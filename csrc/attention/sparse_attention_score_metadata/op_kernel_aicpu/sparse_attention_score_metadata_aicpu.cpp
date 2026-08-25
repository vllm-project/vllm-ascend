/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "log.h"
#include "sparse_attention_score_metadata_aicpu.h"

namespace aicpu {
namespace {

constexpr uint32_t KERNEL_STATUS_OK = 0U;
constexpr uint32_t KERNEL_STATUS_PARAM_INVALID = 1U;

bool HasTensorShape(const Tensor *tensor)
{
    return tensor != nullptr && tensor->GetTensorShape() != nullptr;
}

bool HasTensorData(const Tensor *tensor)
{
    return tensor != nullptr && tensor->GetData() != nullptr;
}

} // namespace

uint32_t GenericBlockSparseAttentionMetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    if (!Prepare(ctx)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    std::vector<int64_t> qSeqLens;
    std::vector<int64_t> qStorageBlockStarts;
    std::vector<int64_t> validBlockNums;
    if (!ParseQSeqLens(qSeqLens, qStorageBlockStarts) ||
        !ParseValidBlockNums(qSeqLens, qStorageBlockStarts, validBlockNums) ||
        !GenerateMetadata(qSeqLens, validBlockNums)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

bool GenericBlockSparseAttentionMetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
    sparseBlockIdx_ = ctx.Input(static_cast<uint32_t>(ParamId::SPARSE_BLOCK_IDX));
    sparseBlockCount_ = ctx.Input(static_cast<uint32_t>(ParamId::SPARSE_BLOCK_COUNT));
    cuSeqLengths_ = ctx.Input(static_cast<uint32_t>(ParamId::CU_SEQ_LENGTHS));
    seqUsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::SEQ_USED_Q));
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::METADATA));
    // An omitted optional input is represented by an empty Tensor in the AICPU context.
    if (!HasTensorData(cuSeqLengths_)) {
        cuSeqLengths_ = nullptr;
    }
    if (!HasTensorData(seqUsedQ_)) {
        seqUsedQ_ = nullptr;
    }

    const bool attrsValid =
        GetAttrValue(ctx, "max_q_seq_len", maxQSeqLen_) && GetAttrValue(ctx, "num_q_heads", numQHeads_) &&
        GetAttrValue(ctx, "num_kv_heads", numKvHeads_) && GetAttrValue(ctx, "head_dim", headDim_) &&
        GetAttrValue(ctx, "block_shape_x", blockShapeX_) && GetAttrValue(ctx, "block_shape_y", blockShapeY_) &&
        GetAttrValue(ctx, "is_packed_gqa", isPackedGQA_) && GetAttrValue(ctx, "q_input_layout", qInputLayout_) &&
        GetAttrValue(ctx, "aic_core_num", aicCoreNum_);
    return attrsValid && CheckInputs();
}

bool GenericBlockSparseAttentionMetadataCpuKernel::CheckOptionalTensor(const Tensor *tensor, DataType dataType,
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

bool GenericBlockSparseAttentionMetadataCpuKernel::CheckInputs()
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
    if (maxQSeqLen_ <= 0 || numQHeads_ <= 0 || numKvHeads_ <= 0 || headDim_ <= 0 || blockShapeX_ != 1 ||
        blockShapeY_ <= 0 || (isPackedGQA_ != 0 && isPackedGQA_ != 1) || aicCoreNum_ <= 0 ||
        aicCoreNum_ > MAX_AIC_CORE_NUM) {
        KERNEL_LOG_ERROR("Invalid scheduling attrs for GenericBlockSparseAttentionMetadata.");
        return false;
    }
    if (qInputLayout_ != "TND" && qInputLayout_ != "BSND" && qInputLayout_ != "BNSD") {
        KERNEL_LOG_ERROR("q_input_layout only supports TND, BSND or BNSD.");
        return false;
    }

    sparseHeadNum_ = isPackedGQA_ == 1 ? numKvHeads_ : numQHeads_;
    const auto idxShape = sparseBlockIdx_->GetTensorShape();
    const auto countShape = sparseBlockCount_->GetTensorShape();
    if (qInputLayout_ == "TND") {
        if (idxShape->GetDims() != 3 || countShape->GetDims() != 2 || idxShape->GetDimSize(0) != sparseHeadNum_ ||
            countShape->GetDimSize(0) != sparseHeadNum_ || countShape->GetDimSize(1) != idxShape->GetDimSize(1)) {
            KERNEL_LOG_ERROR("TND sparse block tensor shapes do not match attrs.");
            return false;
        }
        qBlockStorageNum_ = idxShape->GetDimSize(1);
        blockIndexStride_ = idxShape->GetDimSize(2);
        if (!HasTensorShape(cuSeqLengths_) || cuSeqLengths_->GetTensorShape()->GetDims() != 1) {
            KERNEL_LOG_ERROR("cu_seq_lengths is required for TND query.");
            return false;
        }
        batchSize_ = cuSeqLengths_->GetTensorShape()->GetDimSize(0) - 1;
    } else {
        if (idxShape->GetDims() != 4 || countShape->GetDims() != 3) {
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
        if (idxShape->GetDimSize(2) != countShape->GetDimSize(2)) {
            KERNEL_LOG_ERROR("sparse block Q dims differ: idx=%lld, count=%lld.", idxShape->GetDimSize(2),
                             countShape->GetDimSize(2));
            return false;
        }
        if (idxShape->GetDimSize(2) != maxQSeqLen_) {
            KERNEL_LOG_ERROR("sparse block Q dim must equal max_q_seq_len: Q dim=%lld, max_q_seq_len=%lld.",
                             idxShape->GetDimSize(2), maxQSeqLen_);
            return false;
        }
        batchSize_ = idxShape->GetDimSize(0);
        qBlockStorageNum_ = idxShape->GetDimSize(2);
        blockIndexStride_ = idxShape->GetDimSize(3);
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

bool GenericBlockSparseAttentionMetadataCpuKernel::ParseQSeqLens(std::vector<int64_t> &qSeqLens,
                                                                 std::vector<int64_t> &qStorageBlockStarts) const
{
    // Get valid Qseq len vector. Qseq priroty: seqUsedQ, cuSeqLengths, maxQSeqLen.
    // qSeqLens ：actual Qseqlens in batch. qStorageBlockStarts : Qblocks start(1 qtoken) in batch base on cuSeq(maybe pad).
    if (qInputLayout_ == "TND") {
        const auto *prefix = static_cast<const int64_t *>(cuSeqLengths_->GetData());
        std::vector<int64_t> cuSeqLengths(prefix, prefix + batchSize_ + 1);
        std::vector<int64_t> seqUsedQ;
        if (seqUsedQ_ != nullptr) {
            const auto *seqUsed = static_cast<const int32_t *>(seqUsedQ_->GetData());
            seqUsedQ.assign(seqUsed, seqUsed + batchSize_);
        }
        if (!sparse_attention_score_metadata::BuildTndQSeqLayout(cuSeqLengths, seqUsedQ, seqUsedQ_ != nullptr,
                                                                 maxQSeqLen_, blockShapeX_, qBlockStorageNum_, qSeqLens,
                                                                 qStorageBlockStarts)) {
            KERNEL_LOG_ERROR("TND Q sequence lengths do not match sparse block storage.");
            return false;
        }
        return true;
    }

    qSeqLens.assign(static_cast<size_t>(batchSize_), maxQSeqLen_);
    qStorageBlockStarts.clear();
    if (seqUsedQ_ != nullptr) {
        const auto *seqUsed = static_cast<const int32_t *>(seqUsedQ_->GetData());
        for (int64_t batchIdx = 0; batchIdx < batchSize_; ++batchIdx) {
            const int64_t qSeqLen = seqUsed[batchIdx];
            if (qSeqLen < 0 || qSeqLen > maxQSeqLen_) {
                KERNEL_LOG_ERROR("seq_used_q[%lld]=%lld is outside [0, %lld].", batchIdx, qSeqLen, maxQSeqLen_);
                return false;
            }
            qSeqLens[static_cast<size_t>(batchIdx)] = qSeqLen;
        }
        return true;
    }

    if (cuSeqLengths_ == nullptr) {
        return true;
    }
    const auto *prefix = static_cast<const int64_t *>(cuSeqLengths_->GetData());
    if (prefix[0] != 0) {
        KERNEL_LOG_ERROR("cu_seq_lengths[0] must be 0, but got %lld.", prefix[0]);
        return false;
    }
    for (int64_t batchIdx = 0; batchIdx < batchSize_; ++batchIdx) {
        if (prefix[batchIdx + 1] < prefix[batchIdx]) {
            KERNEL_LOG_ERROR("cu_seq_lengths must be nondecreasing at batch %lld.", batchIdx);
            return false;
        }
        const int64_t qSeqLen = prefix[batchIdx + 1] - prefix[batchIdx];
        if (qSeqLen > maxQSeqLen_) {
            KERNEL_LOG_ERROR("Q sequence length %lld exceeds max_q_seq_len %lld at batch %lld.", qSeqLen, maxQSeqLen_,
                             batchIdx);
            return false;
        }
        qSeqLens[static_cast<size_t>(batchIdx)] = qSeqLen;
    }
    return true;
}

bool GenericBlockSparseAttentionMetadataCpuKernel::ParseValidBlockNums(const std::vector<int64_t> &qSeqLens,
                                                                       const std::vector<int64_t> &qStorageBlockStarts,
                                                                       std::vector<int64_t> &validBlockNums) const
{
    const auto *counts = static_cast<const int32_t *>(sparseBlockCount_->GetData());
    if (qInputLayout_ == "TND") {
        if (!sparse_attention_score_metadata::GatherTndValidBlockNums(counts, sparseHeadNum_, qBlockStorageNum_,
                                                                      blockShapeX_, blockIndexStride_, qSeqLens,
                                                                      qStorageBlockStarts, validBlockNums)) {
            KERNEL_LOG_ERROR("Failed to gather TND sparse block counts.");
            return false;
        }
        return true;
    }

    validBlockNums.clear();
    for (int64_t batchIdx = 0; batchIdx < batchSize_; ++batchIdx) {
        const int64_t qBlockNum =
            sparse_attention_score_metadata::CeilDivQSeq(qSeqLens[static_cast<size_t>(batchIdx)], blockShapeX_);
        for (int64_t qBlock = 0; qBlock < qBlockNum; ++qBlock) {
            for (int64_t sparseHead = 0; sparseHead < sparseHeadNum_; ++sparseHead) {
                const int64_t offset = (batchIdx * sparseHeadNum_ + sparseHead) * qBlockStorageNum_ + qBlock;
                const int64_t count = counts[offset];
                if (count < 0 || count > blockIndexStride_) {
                    KERNEL_LOG_ERROR("sparse_block_count[%lld]=%lld exceeds capacity %lld.", offset, count,
                                     blockIndexStride_);
                    return false;
                }
                validBlockNums.push_back(count);
            }
        }
    }
    return true;
}

bool GenericBlockSparseAttentionMetadataCpuKernel::GenerateMetadata(const std::vector<int64_t> &qSeqLens,
                                                                    const std::vector<int64_t> &validBlockNums)
{
    sparse_attention_score_metadata::ScheduleInput input;
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
    input.aicCoreNum = aicCoreNum_;
    input.qSeqLens = qSeqLens;
    input.validBlockNums = validBlockNums;

    sparse_attention_score_metadata::ScheduleResult result;
    const auto scheduleStatus = sparse_attention_score_metadata::BuildSchedule(input, result);
    if (scheduleStatus != sparse_attention_score_metadata::ScheduleStatus::SUCCESS) {
        KERNEL_LOG_ERROR("Failed to build metadata schedule, status=%u.", static_cast<uint32_t>(scheduleStatus));
        return false;
    }
    auto *metadata =
        static_cast<optiling::generic_block_sparse_attention_metadata::MetadataType *>(metadata_->GetData());
    const auto encodeStatus = sparse_attention_score_metadata::EncodeMetadata(
        result, metadata, optiling::generic_block_sparse_attention_metadata::METADATA_TOTAL_SIZE);
    if (encodeStatus != sparse_attention_score_metadata::ScheduleStatus::SUCCESS) {
        KERNEL_LOG_ERROR("Failed to encode metadata, status=%u.", static_cast<uint32_t>(encodeStatus));
        return false;
    }
    return true;
}

namespace {
const char *KERNEL_TYPE = "GenericBlockSparseAttentionMetadata";
REGISTER_CPU_KERNEL(KERNEL_TYPE, GenericBlockSparseAttentionMetadataCpuKernel);
} // namespace

} // namespace aicpu
