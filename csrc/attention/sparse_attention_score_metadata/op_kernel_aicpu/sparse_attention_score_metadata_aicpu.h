/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef SPARSE_ATTENTION_SCORE_METADATA_AICPU_H
#define SPARSE_ATTENTION_SCORE_METADATA_AICPU_H

#include <cstdint>
#include <string>
#include <vector>

#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"
#include "../../common/op_kernel/aicpu_common.h"
#include "sparse_attention_score_metadata_q_seq_utils.h"
#include "sparse_attention_score_metadata_scheduler.h"

namespace aicpu {

class GenericBlockSparseAttentionMetadataCpuKernel : public CpuKernel {
public:
    GenericBlockSparseAttentionMetadataCpuKernel() = default;
    ~GenericBlockSparseAttentionMetadataCpuKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    bool Prepare(CpuKernelContext &ctx);
    bool CheckInputs();
    bool ParseQSeqLens(std::vector<int64_t> &qSeqLens, std::vector<int64_t> &qStorageBlockStarts) const;
    bool ParseValidBlockNums(const std::vector<int64_t> &qSeqLens, const std::vector<int64_t> &qStorageBlockStarts,
                             std::vector<int64_t> &validBlockNums) const;
    bool GenerateMetadata(const std::vector<int64_t> &qSeqLens, const std::vector<int64_t> &validBlockNums);
    bool CheckOptionalTensor(const Tensor *tensor, DataType dataType, int64_t elementNum, const char *tensorName) const;

    Tensor *sparseBlockIdx_ = nullptr;
    Tensor *sparseBlockCount_ = nullptr;
    Tensor *cuSeqLengths_ = nullptr;
    Tensor *seqUsedQ_ = nullptr;
    Tensor *metadata_ = nullptr;

    int64_t batchSize_ = 0;
    int64_t maxQSeqLen_ = 0;
    int64_t numQHeads_ = 0;
    int64_t numKvHeads_ = 0;
    int64_t headDim_ = 0;
    int64_t blockShapeX_ = 1;
    int64_t blockShapeY_ = 0;
    int64_t isPackedGQA_ = 1;
    int64_t aicCoreNum_ = 0;
    int64_t sparseHeadNum_ = 0;
    int64_t qBlockStorageNum_ = 0; // QShape nums of SparseBlockCount last dim.
    int64_t blockIndexStride_ = 0; // maxSparseBlockCount : the max value of valid blocks.
    std::string qInputLayout_;

    enum class ParamId : uint32_t {
        SPARSE_BLOCK_IDX = 0U,
        SPARSE_BLOCK_COUNT = 1U,
        CU_SEQ_LENGTHS = 2U,
        CU_SEQ_LENGTHS_KV = 3U,
        SEQ_USED_Q = 4U,
        SEQ_USED_KV = 5U,
        METADATA = 0U,
    };
};

} // namespace aicpu

#endif // SPARSE_ATTENTION_SCORE_METADATA_AICPU_H
