/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file paged_attention_antiquantkv.h
 * \brief
 */

#ifndef PAGED_ATTENTION_ANTIQUANTKV_H
#define PAGED_ATTENTION_ANTIQUANTKV_H

#include "common.h"
#include "common_func.h"
#include "simd.h"
#include "iterator.h"
#include "mma.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../ifa_public_define.h"
#include "gm_to_l1_iterator.h"
#include "gm_to_ub_iterator.h"
#include "l0c_to_gm_iterator.h"
#include "l1_to_l0_iterator.h"

// define common const value
// FFTS Flag
constexpr uint32_t QK_READY_FLAG = 2;
constexpr uint32_t SOFTMAX_READY_D = 4;
constexpr uint32_t UPDATE_READY_D = 6;
constexpr uint32_t VEC_DEQ_K0_READY = 9;
constexpr uint32_t VEC_DEQ_K1_READY = 10;
constexpr uint32_t VEC_DEQ_V0_READY = 11;
constexpr uint32_t VEC_DEQ_V1_READY = 12;

constexpr int32_t BLOCK_SIZE_D = 16;
constexpr int64_t TMP_SIZE_64K = 65536;

const int32_t TILING_NUMHEADS = 1;
const int32_t TILING_HEADDIM = 2;
const int32_t TILING_NUMBLOKS = 3;
const int32_t TILING_BLOCKSIZE = 4;
const int32_t TILING_MAXBLOCKS = 5;
const int32_t TILING_TOR = 6;
const int32_t TILING_KVHEADS = 7;
const int32_t TILING_FORMER_BATCH = 8;
const int32_t TILING_FORMER_HEAD = 9;
const int32_t TILING_TAIL_BATCH = 10;
const int32_t TILING_TAIL_HEAD = 11;
const int32_t TILING_HEADNUM_MOVE = 12;
const int32_t TILING_MASK_MAX_LEN = 13;
const int32_t TILING_BATCH_STRIDE = 14;
const int32_t TILING_HEAD_STRIDE = 15;
const int32_t TILING_KEY = 16;
const int32_t TILING_HEADSIZE = 17;
const int32_t TILING_PARASIZE = 18;
const int32_t TILING_GROUPNUM = 19;
const int32_t TILING_FORMER_GROUP_MOVE = 20;
const int32_t TILING_TAIL_GROUP_MOVE = 21;
const int32_t TILING_MAX_KVSEQLEN = 22;
const int32_t TILING_KVSPLIT = 23;
const int32_t TILING_KVCORENUM = 24;
const int32_t TILING_BLOCKSIZE_CALC = 25;
const int32_t TILING_TOTAL_BLOCK_NUM = 26;
const int32_t TILING_PREFILL_BS = 27;
const int32_t TILING_DECODER_BS = 28;
const int32_t TILING_HEADDIM_V = 29;
const int32_t TILING_MODCOEF = 30;
const int32_t TILING_DIVCOEF = 31;
const int32_t TILING_QHEADORIGINAL = 32;
const int32_t TILING_COMPRESSHEAD = 33;
const int32_t TILING_QUANTYPE = 34;
const int32_t TILING_DATA_SHAPE_TYPE = 35;
const int32_t TILING_SCALETYPE = 36;
const int32_t TILING_MASK_TYPE_ND = 37;
const int32_t TILING_HEADDIM_K_SPLIT = 38;
const int32_t TILING_HEADDIM_V_SPLIT = 39;
const int32_t TILING_HEADDIM_V_SPLIT_VECTOR_FORMER = 40;
const int32_t TILING_HEADDIM_V_SPLIT_VECTOR_TAIL = 41;
const int32_t BLOCKSIZE_CALC_256 = 256;
constexpr uint32_t CONST_16 = 16;
constexpr uint32_t KV_SEQ_STEP = 16;
constexpr uint32_t MAX_NUMEL_INST_B8 = 255 * 256;
constexpr uint32_t MAX_NUMEL_INST_B16 = 255 * 128;
constexpr uint32_t MAX_NUMEL_INST_B32 = 255 * 64;

constexpr int32_t L0AB_HALF_BUF_SIZE_D = 16384;
constexpr int32_t L0AB_UINT8_BUF_SIZE = 32768;
constexpr int32_t L0C_FLOAT_BUF_SIZE = 16384;
constexpr int32_t L0C_UINT8_BUF_SIZE = 131072;
constexpr int32_t CUBE_MATRIX_SIZE_D = 256;
constexpr int64_t L0AB_UINT8_BLOCK_SIZE_D = 32768;
constexpr int32_t L1_HALF_BUF_SIZE = 65536;
constexpr int32_t L1_P_UINT8_BUF_SIZE = 32768;
constexpr int32_t TMP_SIZE_DECODER = 32768;
constexpr int32_t L1_HALF_BUF_SIZE_DECODER = 16384;
constexpr int32_t L1_UINT8_BUF_SIZE_DECODER = 32768;
constexpr int32_t L1_KV_HALF_BUF_SIZE = 65536;
constexpr int32_t L1_KV_UINT8_BUF_SIZE = 65536 * 2;
constexpr uint64_t L1_E_UINT8_SIZE = 1024;
constexpr uint64_t L1_SCALE_UINT8_SIZE = 4096;
constexpr uint64_t L1_SCALE_UINT64_SIZE = 512;
constexpr uint64_t L1_OFFSET_UINT8_SIZE = 2048;
constexpr uint64_t L1_OFFSET_INT32_SIZE = 512;

constexpr uint32_t L0AB_PINGPONG_BUFFER_LEN = 32768;
constexpr uint32_t L0C_PINGPONG_BUFFER_LEN_INT32 = 16384;
constexpr uint32_t CUBE_MATRIX_SIZE_512 = 512;
constexpr int32_t BLOCK_SIZE_16 = 16;
constexpr uint64_t CONST_4 = 4;
constexpr uint64_t CONST_8 = 8;
constexpr uint64_t CONST_32 = 32;
constexpr uint64_t CONST_64 = 64;
constexpr uint64_t CONST_128 = 128;
constexpr uint32_t EMBED_SPLIT = 256;
constexpr uint32_t ROUND_EMBED_SPLIT = 256;

constexpr uint32_t HALF_VECTOR_SIZE = 128;
constexpr uint32_t UB_ALIGN_BYTE = 32;
constexpr int32_t FLOAT_VECTOR_SIZE_D = 64;
constexpr int64_t UB_UINT8_BLOCK_SIZE_MLA = 16384;
constexpr int64_t UB_UINT8_BLOCK_SIZE_NORM = 24576;
constexpr int64_t UB_UINT8_LINE_SIZE_D = 512;
constexpr int64_t UB_HALF_LINE_SIZE_D = 256;
constexpr int64_t UB_FLOAT_LINE_SIZE_D = 128;

constexpr int64_t PRE_UB_UINT8_BLOCK_SIZE = 16384;
constexpr int32_t VECTOR_SIZE_D = 128;
constexpr int32_t FLOAT_BLOCK_SIZE_D = 8;
constexpr int32_t UB_HALF_BUF_SIZE_D = 8192;
constexpr int32_t STAGE2_UB_UINT8_BLOCK_SIZE = 8192;
constexpr uint32_t MAX_UB_SIZE = 196608;
constexpr uint32_t EMBED_SPLIT_SM = 128;
constexpr uint32_t ROUND_EMBED_SPLIT_SM = 128;
constexpr uint32_t TASK_ROUND_TWO = 2;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t AIC_AIV_RATIO = 2;
constexpr uint32_t FLOAT_BYTES = 4;
constexpr uint32_t HALF_BYTES = 2;
constexpr uint32_t SYNC_AIC_AIV_MODE = 2;
constexpr uint32_t BLOCK_NUM_IN_REPEAT = 8;
constexpr uint32_t BLOCK_NUM_IN_REPEAT_HALF = 4;
constexpr uint32_t VEC_REPEAT_NUM_255 = 255;
constexpr uint32_t BLK_STRIDE_ONE = 1;
constexpr uint32_t MASK_TYPE_3 = 3;
constexpr uint64_t CONST_TWO = 2;
constexpr uint64_t CONST_6 = 6;
constexpr uint64_t CONST_9 = 9;
constexpr uint64_t CONST_11 = 11;
constexpr uint64_t CONST_17 = 17;

enum class TilingKeyType {
    TILING_HALF_DATA = 0,
    TILING_BF16_DATA = 1,
    TILING_INT8_DATA = 2,
    TILING_INT8_CUBE_QUANT = 4,
    TILING_INT8_VEC_QUANT = 8,
    TILING_INT8_VEC_QUANTBF16 = 9,
    TILING_QUANT_FP16OUT = 12,
    TILING_QUANT_BF16OUT = 14
};

template<TilingKeyType tilingKeyType>
struct AttentionType
{
};

// pa Parallel feature class 
template <typename IFAT, TilingKeyType tilingKeyType = TilingKeyType::TILING_INT8_VEC_QUANT>
class PagedAttentionParallelAic {
public:
    using Q_T = typename IFAT::queryType;
    using KV_T = typename IFAT::queryType;
    using IN_DATA_TYPE = typename IFAT::queryType;
    using S_DATA_TYPE = half;
    using TILING_T = typename IFAT::TilingType;

    __aicore__ inline PagedAttentionParallelAic(uint32_t prefill_batch_size, uint32_t decoder_batch_size) {
        prefill_batch_size_ = prefill_batch_size;
        decoder_batch_size_ = decoder_batch_size;
    }

    __aicore__ inline void SetArgs(
        __gm__ uint8_t *__restrict__ q_in_gm,
        __gm__ uint8_t *__restrict__ k_in_gm,
        __gm__ uint8_t *__restrict__ v_in_gm,
        __gm__ uint8_t *__restrict__ seq_len_q_gm,
        __gm__ uint8_t *__restrict__ seq_len_kv_gm,
        __gm__ uint8_t *__restrict__ block_tables_in_gm,
        __gm__ uint8_t *__restrict__ workspace_gm,
        const typename IFAT::TilingType *__restrict tiling) {

        q_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ Q_T *>(q_in_gm));

        actualSeqLenGmTensorQ.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(seq_len_q_gm));
        actualSeqLenGmTensorKv.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(seq_len_kv_gm));
        blockTableGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(block_tables_in_gm));

        const uint32_t l1q_buf_addr_offset = 0;
        const uint32_t l1k_buf_addr_offset = DOUBLE_BUFFER_NUM * L0AB_UINT8_BLOCK_SIZE_D;
        const uint32_t l1p_buf_addr_offset = l1k_buf_addr_offset + DOUBLE_BUFFER_NUM * L0AB_UINT8_BLOCK_SIZE_D;
        const uint32_t l1v_buf_addr_offset = l1p_buf_addr_offset + DOUBLE_BUFFER_NUM * L0AB_UINT8_BLOCK_SIZE_D;

        l1q_buf_addr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, Q_T>(l1q_buf_addr_offset);
        l1k_buf_addr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, KV_T>(l1k_buf_addr_offset);
        l1p_buf_addr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, Q_T>(l1p_buf_addr_offset);
        l1v_buf_addr_tensor = buf.GetBuffer<BufferType::ASCEND_CB, KV_T>(l1v_buf_addr_offset);

        l0a_buf_tensor = buf.GetBuffer<BufferType::ASCEND_L0A, Q_T>(0);
        l0b_buf_tensor = buf.GetBuffer<BufferType::ASCEND_L0B, KV_T>(0);
        l0c_buf_tensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>(0);

        batchSize_ = tiling->tilingBase.batchSize;
        qHeadNum_ = tiling->tilingBase.qHeadNum;
        embedding_size = tiling->tilingBase.headSize;
        kvHeadNum_ = tiling->tilingBase.kvHeadNum;
        maxNumBlocksPerQuery = tiling->tilingBase.maxBlockNumPerBatch;
        kvBlockSize_ = tiling->tilingBase.blockSize;
        mask_type = tiling->tilingBase.attenMaskFlag;
        totalBlockNumQ_ = tiling->tilingPerCore.totalQBlockNum;
        seqStepQ_ = tiling->tilingPerCore.seqStepQ;
        seqStepKv_ = tiling->tilingPerCore.seqStepKv;

        groupNum = qHeadNum_ / kvHeadNum_;
        stride_kv = static_cast<uint64_t>(kvHeadNum_) * embedding_size;
        stride_qo = static_cast<uint64_t>(qHeadNum_) * embedding_size;

        headDim_ = embedding_size;
        round_k = (headDim_ + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16 * BLOCK_SIZE_16;

        numBlocks_ = GetBlockNum();
        blockIdx_ = GetBlockIdx();

        uint32_t dequantKeyPingOffset = 0;
        // 每一次只涉及一个qHead和kvHead, 所以*1
        uint32_t dequantKeyPongOffset = dequantKeyPingOffset + numBlocks_ * BLOCKSIZE_CALC_256 * 1 * embedding_size * sizeof(KV_T);
        uint32_t dequantValuePingOffset = dequantKeyPongOffset + numBlocks_ * BLOCKSIZE_CALC_256 * 1 * embedding_size * sizeof(KV_T);
        uint32_t dequantValuePongOffset = dequantValuePingOffset + numBlocks_ * BLOCKSIZE_CALC_256 * 1 * embedding_size * sizeof(KV_T);
        uint32_t scoreWsOffset = dequantValuePongOffset + numBlocks_ * BLOCKSIZE_CALC_256 * 1 * embedding_size * sizeof(KV_T);
        uint32_t probWsOffset = scoreWsOffset + numBlocks_ * TMP_SIZE_64K * sizeof(S_DATA_TYPE);
        uint32_t outputTmpOffset = probWsOffset + numBlocks_ * TMP_SIZE_64K * sizeof(IN_DATA_TYPE);

        dequantKeyWsPing_.SetGlobalBuffer(reinterpret_cast<__gm__ KV_T*>(workspace_gm + dequantKeyPingOffset));
        dequantKeyWsPong_.SetGlobalBuffer(reinterpret_cast<__gm__ KV_T*>(workspace_gm + dequantKeyPongOffset));
        dequantValueWsPing_.SetGlobalBuffer(reinterpret_cast<__gm__ KV_T*>(workspace_gm + dequantValuePingOffset));
        dequantValueWsPong_.SetGlobalBuffer(reinterpret_cast<__gm__ KV_T*>(workspace_gm + dequantValuePongOffset));

        s_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ S_DATA_TYPE *>(workspace_gm + scoreWsOffset));
        p_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ Q_T *>(workspace_gm + probWsOffset));
        o_tmp_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_gm + outputTmpOffset));
    }

    __aicore__ inline void Run() {
        SET_FLAG(MTE1, MTE2, EVENT_ID0);
        SET_FLAG(MTE1, MTE2, EVENT_ID1);
        SET_FLAG(MTE1, MTE2, EVENT_ID2);
        SET_FLAG(MTE1, MTE2, EVENT_ID3);
        SET_FLAG(M, MTE1, EVENT_ID0);
        SET_FLAG(M, MTE1, EVENT_ID1);
        SET_FLAG(FIX, M, EVENT_ID0);
        SET_FLAG(FIX, M, EVENT_ID1);

        uint32_t curBatchIdx = 0;
        // kvSeqLen相关数据
        uint32_t preAccumSeqLenKv = 0;
        uint32_t accumSeqLenKv = actualSeqLenGmTensorKv.GetValue(curBatchIdx);
        uint32_t kvSeqLenCurBatch = accumSeqLenKv - preAccumSeqLenKv;

        // kvSeqQ相关数据
        uint32_t preAccumBlkNumQ = 0;
        uint32_t preAccumSeqLenQ = 0;
        uint32_t accumSeqLengthQ = actualSeqLenGmTensorQ.GetValue(curBatchIdx);
        uint32_t qSeqLenCurBatch = accumSeqLengthQ - preAccumSeqLenQ;
        uint32_t accumQueryBlkNum = preAccumBlkNumQ + (qSeqLenCurBatch + seqStepQ_ - 1) / seqStepQ_;

        uint32_t totalProcessNum = totalBlockNumQ_ * qHeadNum_;
        for (uint32_t process = 0; process < totalProcessNum; process++) {
            if (process >= accumQueryBlkNum * qHeadNum_) {
                bool doRunLoopFlag = true;
                while (doRunLoopFlag) {
                    curBatchIdx++;
                    // 记录上一次的累计数据
                    preAccumBlkNumQ = accumQueryBlkNum;
                    preAccumSeqLenQ = accumSeqLengthQ;
                    preAccumSeqLenKv = accumSeqLenKv;

                    accumSeqLenKv = actualSeqLenGmTensorKv.GetValue(curBatchIdx);
                    kvSeqLenCurBatch = accumSeqLenKv - preAccumSeqLenKv;
                    accumSeqLengthQ = actualSeqLenGmTensorQ.GetValue(curBatchIdx);
                    qSeqLenCurBatch = accumSeqLengthQ - preAccumSeqLenQ;
                    // 必须逐步累加，不能用Q的前序和计算，会有batch的尾块问题
                    accumQueryBlkNum += (qSeqLenCurBatch + seqStepQ_ - 1) / seqStepQ_;
                    if(qSeqLenCurBatch != 0) {
                        break;
                    }

                    if (curBatchIdx >= batchSize_ - 1) {
                        doRunLoopFlag = false;
                        break;
                    }
                }
            }
            uint32_t targetCoreIdx = process % numBlocks_;
            if (is_triu_mask) {
                if ((process / numBlocks_) % TASK_ROUND_TWO == 1) { // 第二轮任务: 任务顺序反转一下
                    targetCoreIdx = numBlocks_ - process % numBlocks_ - 1;
                }
            }
            if (blockIdx_ != targetCoreIdx) { // core 0, cur_core_idx其他任务跳过
                continue;
            }

            uint64_t queryOffsetBase = preAccumSeqLenQ * qHeadNum_ * embedding_size;

            uint32_t process_idx = process - preAccumBlkNumQ * qHeadNum_;
            uint32_t m_idx = process_idx / qHeadNum_;
            uint32_t head_idx = process_idx % qHeadNum_;
            uint32_t kvHeadIdx = head_idx / groupNum;

            uint32_t m_loop = (qSeqLenCurBatch + seqStepQ_ - 1) / seqStepQ_;
            uint32_t n_loop = (kvSeqLenCurBatch + seqStepKv_ - 1) / seqStepKv_;  // seqStepKv_ != kvBlockSize_ 处理

            uint32_t qk_m = (m_idx == (m_loop - 1)) ? (qSeqLenCurBatch - m_idx * seqStepQ_) : seqStepQ_;
            uint32_t qk_round_m = (qk_m + BLOCK_SIZE_D - 1) / BLOCK_SIZE_D * BLOCK_SIZE_D;

            uint64_t qk_index = 0;
            /**************** pre_load *****************/
            uint32_t qk_n = (qk_index == (n_loop - 1)) ? kvSeqLenCurBatch : seqStepKv_;
            uint32_t qk_round_n = (qk_n + BLOCK_SIZE_D - 1) / BLOCK_SIZE_D * BLOCK_SIZE_D;

            uint32_t pingpong_flag = 0;
            uint32_t offset = pingpong_flag * L0AB_HALF_BUF_SIZE_D;
            uint64_t q_offset = queryOffsetBase + head_idx * embedding_size + m_idx * seqStepQ_ * stride_qo;

            uint32_t sv_n = seqStepKv_;
            uint32_t sv_round_n = (sv_n + BLOCK_SIZE_D - 1) / BLOCK_SIZE_D * BLOCK_SIZE_D;
            uint32_t n_end = n_loop;
            if (is_triu_mask) {
                uint32_t triu_seq = mask_type == MASK_TYPE_3 ? max_kv - qSeqLenCurBatch : kvSeqLenCurBatch - qSeqLenCurBatch;
                uint32_t n_offset = ((m_idx + 1) * seqStepQ_ + triu_seq + seqStepKv_ - 1) / seqStepKv_;
                n_end = n_offset > n_loop ? n_loop : n_offset;
            }

            // Only need load Q once
            // gm_to_l1q
            if (qk_m == 1) {
                gm_to_l1<ArchType::ASCEND_V220, Q_T, DataFormatT::ND, DataFormatT::ND>(
                    l1q_buf_addr_tensor,
                    q_gm_tensor[q_offset],
                    1,
                    0,
                    0,
                    round_k,               // lenBurst
                    0,
                    0
                );
            } else {
                gm_to_l1<ArchType::ASCEND_V220, Q_T, DataFormatT::ND, DataFormatT::NZ>(
                    l1q_buf_addr_tensor,
                    q_gm_tensor[q_offset],
                    qk_m,        // nValue
                    qk_round_m,  // dstNzC0Stride
                    0,           // dstNzMatrixStride, unused
                    headDim_,         // dValue
                    0,           // dstNzMatrixStride, unused
                    stride_qo    // srcDValue
                );
            }
            SET_FLAG(MTE2, MTE1, pingpong_flag);
            WAIT_FLAG(MTE2, MTE1, pingpong_flag);
            uint32_t KV_INC = 1;
            uint32_t vect_mod = AIC_AIV_RATIO * KV_INC;
            for (uint32_t n_idx = 0; n_idx < n_end + KV_INC; n_idx += KV_INC) {
                if (n_idx < n_end) {
                    for (uint32_t split_idx = 0; split_idx < KV_INC && n_idx + split_idx < n_end; split_idx++) {
                        int64_t k_offset = blockIdx_ * BLOCKSIZE_CALC_256 * 1 * embedding_size;
                        SET_FLAG(S, MTE2, EVENT_ID0);

                        pingpong_flag = (n_idx + split_idx) % DOUBLE_BUFFER_NUM;
                        offset = pingpong_flag * L0AB_HALF_BUF_SIZE_D;
                        if (n_idx + split_idx == (n_loop - 1)) {
                            qk_n = (kvSeqLenCurBatch - (n_idx + split_idx) * seqStepKv_);
                            qk_round_n = (qk_n + BLOCK_SIZE_D - 1) / BLOCK_SIZE_D * BLOCK_SIZE_D;
                        }
                        WAIT_FLAG(M, MTE1, pingpong_flag);
                        if (qk_m == 1) {
                            l1_to_l0_a<ArchType::ASCEND_V220, Q_T, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                                l0a_buf_tensor[offset],
                                l1q_buf_addr_tensor,
                                0,
                                (round_k + CUBE_MATRIX_SIZE_D - 1) / CUBE_MATRIX_SIZE_D,  // repeat
                                0,
                                1,                                                    // srcStride
                                0,
                                0                                                    // dstStride
                            );
                        } else {
                            for (uint32_t l0a_load_idx = 0; l0a_load_idx < qk_round_m / BLOCK_SIZE_D; ++l0a_load_idx) {
                                l1_to_l0_a<ArchType::ASCEND_V220, Q_T, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                                    l0a_buf_tensor[offset + l0a_load_idx * round_k * BLOCK_SIZE_D],
                                    l1q_buf_addr_tensor[l0a_load_idx * CUBE_MATRIX_SIZE_D],
                                    0,
                                    round_k / BLOCK_SIZE_D,     // repeat
                                    0,
                                    qk_round_m / BLOCK_SIZE_D,  // srcStride
                                    0,
                                    0                        // dstStride
                                );
                            }
                        }
                        // *** Prepare K to L1
                        WAIT_FLAG(S, MTE2, EVENT_ID0);
                        WAIT_FLAG(MTE1, MTE2, pingpong_flag);
                        if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_VEC_QUANT || tilingKeyType == TilingKeyType::TILING_INT8_VEC_QUANTBF16) {
                            uint32_t dequantKeySyncFlag = VEC_DEQ_K0_READY + pingpong_flag;
                            AscendC::CrossCoreWaitFlag(dequantKeySyncFlag);

                            uint32_t dequantValueSyncFlag = VEC_DEQ_V0_READY + pingpong_flag;
                            AscendC::CrossCoreWaitFlag(dequantValueSyncFlag);
                        }
                        // gm_to_l1k
                        AscendC::GlobalTensor<KV_T> dequantKeyWs = pingpong_flag ? dequantKeyWsPong_ : dequantKeyWsPing_;
                        gm_to_l1<ArchType::ASCEND_V220, KV_T, DataFormatT::ND, DataFormatT::NZ>(
                            l1k_buf_addr_tensor[offset],
                            dequantKeyWs[k_offset],
                            qk_n,        // nValue
                            qk_round_n,  // dstNzC0Stride
                            0,            // dstNzMatrixStride, unused
                            headDim_,         // dValue
                            0,            // dstNzMatrixStride, unused
                            headDim_   // srcDValue
                        );

                        SET_FLAG(MTE2, MTE1, pingpong_flag);
                        WAIT_FLAG(MTE2, MTE1, pingpong_flag);
                        l1_to_l0_b<ArchType::ASCEND_V220, KV_T, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                            l0b_buf_tensor[offset],
                            l1k_buf_addr_tensor[offset],
                            0,
                            round_k * qk_round_n / CUBE_MATRIX_SIZE_D,  // repeat
                            0,
                            1,                                        // srcStride
                            0,
                            0                                        // dstStride
                        );
                        SET_FLAG(MTE1, MTE2, pingpong_flag);
                        SET_FLAG(MTE1, M, pingpong_flag);
                        WAIT_FLAG(MTE1, M, pingpong_flag);
                        WAIT_FLAG(FIX, M, pingpong_flag);
                        mmad<ArchType::ASCEND_V220, Q_T, KV_T, float, false>(
                            l0c_buf_tensor[offset],
                            l0a_buf_tensor[offset],
                            l0b_buf_tensor[offset],
                            qk_m,  // m
                            qk_n,  // n
                            headDim_,   // k
                            1      // cmatrixInitVal
                        );
                        SET_FLAG(M, MTE1, pingpong_flag);
                        PIPE_BARRIER(ALL);
                        // copy S to gm
                        l0c_to_gm<ArchType::ASCEND_V220, DataFormatT::ND, S_DATA_TYPE, float>(
                            s_gm_tensor[(uint64_t)blockIdx_ * TMP_SIZE_64K + (n_idx + split_idx) % vect_mod * TMP_SIZE_64K / vect_mod],
                            l0c_buf_tensor[offset],
                            qk_m,        // MSize
                            qk_round_n,  // NSize
                            qk_round_m,  // srcStride
                            qk_round_n  // dstStride_dst_D
                        );
                        SET_FLAG(FIX, M, pingpong_flag);
                    }
                    if ASCEND_IS_AIC {
                        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(QK_READY_FLAG);
                    }
                }
                uint32_t sv_n_pre = sv_n;
                uint32_t sv_round_n_pre = sv_round_n;
                uint32_t dequant_pingpong_flag = 0;
                for (uint32_t split_idx = 0; split_idx < KV_INC && n_idx + split_idx < n_end + KV_INC; split_idx++) {
                    pingpong_flag = (n_idx + split_idx) % DOUBLE_BUFFER_NUM;
                    dequant_pingpong_flag = (n_idx + split_idx + KV_INC) % DOUBLE_BUFFER_NUM;
                    offset = pingpong_flag * L0AB_HALF_BUF_SIZE_D;
                    if (n_idx + split_idx >= KV_INC) {
                        if (n_idx + split_idx == (n_loop + KV_INC - 1)) {
                            sv_n = (kvSeqLenCurBatch - (n_idx + split_idx - KV_INC) * seqStepKv_);
                            sv_round_n = (sv_n + BLOCK_SIZE_D - 1) / BLOCK_SIZE_D * BLOCK_SIZE_D;
                        }
                        int64_t v_offset = blockIdx_ * BLOCKSIZE_CALC_256 * 1 * embedding_size; 

                        SET_FLAG(S, MTE2, EVENT_ID0);
                        // *** Prepare V to L1
                        WAIT_FLAG(MTE1, MTE2, pingpong_flag + DOUBLE_BUFFER_NUM);
                        WAIT_FLAG(S, MTE2, EVENT_ID0);

                        //gm_to_l1v
                        AscendC::GlobalTensor<KV_T> dequantValueWs = dequant_pingpong_flag ? dequantValueWsPong_ : dequantValueWsPing_;
                        gm_to_l1<ArchType::ASCEND_V220, KV_T, DataFormatT::ND, DataFormatT::NZ>(
                            l1v_buf_addr_tensor[offset],
                            dequantValueWs[v_offset],
                            sv_n,        // nValue
                            sv_round_n,  // dstNzC0Stride
                            0,            // dstNzMatrixStride, unused
                            headDim_,         // dValue
                            0,            // dstNzMatrixStride, unused
                            headDim_   // srcDValue
                        );
                        SET_FLAG(MTE2, MTE1, pingpong_flag);
                        WAIT_FLAG(MTE2, MTE1, pingpong_flag);
                        WAIT_FLAG(M, MTE1, pingpong_flag);
                        for (uint32_t l0b_load_idx = 0; l0b_load_idx < sv_round_n / BLOCK_SIZE_D; ++l0b_load_idx) {
                            l1_to_l0_b<ArchType::ASCEND_V220, KV_T, true, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                                l0b_buf_tensor[offset + l0b_load_idx * round_k * BLOCK_SIZE_D],
                                l1v_buf_addr_tensor[offset + l0b_load_idx * CUBE_MATRIX_SIZE_D],
                                0,
                                round_k / BLOCK_SIZE_D,     // repeat
                                0,
                                sv_round_n / BLOCK_SIZE_D,  // srcStride
                                0,
                                0                        // dstStride
                            );
                        }
                    }
                }
                uint32_t sv_n = sv_n_pre;
                uint32_t sv_round_n = sv_round_n_pre;
                if (n_idx >= KV_INC) {
                    if ASCEND_IS_AIC {
                        AscendC::CrossCoreWaitFlag(SOFTMAX_READY_D);
                    }
                }
                for (uint32_t split_idx = 0; split_idx < KV_INC && n_idx + split_idx < n_end + KV_INC; split_idx++) {
                    pingpong_flag = (n_idx + split_idx) % DOUBLE_BUFFER_NUM;
                    offset = pingpong_flag * L0AB_HALF_BUF_SIZE_D;
                    if (n_idx + split_idx >= KV_INC) {
                        if (n_idx + split_idx == (n_loop + KV_INC - 1)) {
                            sv_n = (kvSeqLenCurBatch - (n_idx + split_idx - KV_INC) * seqStepKv_);
                            sv_round_n = (sv_n + BLOCK_SIZE_D - 1) / BLOCK_SIZE_D * BLOCK_SIZE_D;
                        }
                        // *** Prepare P to L1
                        if (qk_m == 1) {
                            gm_to_l1<ArchType::ASCEND_V220, Q_T, DataFormatT::ND, DataFormatT::ND>(
                                l1p_buf_addr_tensor[offset],
                                p_gm_tensor[(uint64_t)blockIdx_ * TMP_SIZE_64K + (n_idx + split_idx + KV_INC) % vect_mod * TMP_SIZE_64K / vect_mod],
                                1,
                                0,
                                0,
                                sv_round_n,               // lenBurst
                                0,
                                0
                            );
                        } else {
                            gm_to_l1<ArchType::ASCEND_V220, Q_T, DataFormatT::ND, DataFormatT::NZ>(
                                l1p_buf_addr_tensor[offset],
                                p_gm_tensor[(uint64_t)blockIdx_ * TMP_SIZE_64K + (n_idx + split_idx + KV_INC) % vect_mod * TMP_SIZE_64K / vect_mod],
                                qk_m,        // nValue
                                qk_round_m,  // dstNzC0Stride
                                0,            // dstNzMatrixStride, unused
                                sv_n,        // dValue
                                0,            // dstNzMatrixStride, unused
                                sv_round_n  // srcDValue
                            );
                        }
                        SET_FLAG(MTE2, MTE1, pingpong_flag);
                        WAIT_FLAG(MTE2, MTE1, pingpong_flag);
                        if (qk_m == 1) {
                            l1_to_l0_a<ArchType::ASCEND_V220, Q_T, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                                l0a_buf_tensor[offset],
                                l1p_buf_addr_tensor[offset],
                                0,
                                (sv_round_n + CUBE_MATRIX_SIZE_D - 1) / CUBE_MATRIX_SIZE_D,  // repeat
                                0,
                                1,                                                       // srcStride
                                0,
                                0                                                       // dstStride
                            );
                        } else {
                            for (uint32_t l0a_load_idx = 0; l0a_load_idx < qk_round_m / BLOCK_SIZE_D; ++l0a_load_idx) {
                                l1_to_l0_a<ArchType::ASCEND_V220, Q_T, false, DataFormatT::VECTOR, DataFormatT::VECTOR>(
                                    l0a_buf_tensor[offset + l0a_load_idx * sv_round_n * BLOCK_SIZE_D],
                                    l1p_buf_addr_tensor[offset + l0a_load_idx * CUBE_MATRIX_SIZE_D],
                                    0,
                                    sv_round_n / BLOCK_SIZE_D,  // repeat
                                    0,
                                    qk_round_m / BLOCK_SIZE_D,  // srcStride
                                    0,
                                    0                        // dstStride
                                );
                            }
                        }
                        SET_FLAG(MTE1, MTE2, pingpong_flag + DOUBLE_BUFFER_NUM);
                        SET_FLAG(MTE1, M, pingpong_flag);
                        WAIT_FLAG(MTE1, M, pingpong_flag);
                        WAIT_FLAG(FIX, M, pingpong_flag);
                        mmad<ArchType::ASCEND_V220, Q_T, KV_T, float, false>(
                            l0c_buf_tensor[offset],
                            l0a_buf_tensor[offset],
                            l0b_buf_tensor[offset],
                            qk_m,  // m
                            headDim_,   // n
                            sv_n,  // k
                            1      // cmatrixInitVal
                        );
                        SET_FLAG(M, MTE1, pingpong_flag);
                        PIPE_BARRIER(ALL);
                        // copy O to gm
                        l0c_to_gm<ArchType::ASCEND_V220, DataFormatT::ND, float, float>(
                            o_tmp_gm_tensor[(uint64_t)blockIdx_ * TMP_SIZE_64K + (n_idx + split_idx + KV_INC) % vect_mod * TMP_SIZE_64K / vect_mod],
                            l0c_buf_tensor[offset],
                            qk_m,        // MSize
                            round_k,     // NSize
                            qk_round_m,  // srcStride
                            round_k     // dstStride_dst_D
                        );
                        SET_FLAG(FIX, M, pingpong_flag);
                    }
                }
                if (n_idx >= KV_INC) {
                    if ASCEND_IS_AIC {
                        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(UPDATE_READY_D);
                    }
                }
            }
        }
        WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
        WAIT_FLAG(M, MTE1, EVENT_ID0);
        WAIT_FLAG(M, MTE1, EVENT_ID1);
        WAIT_FLAG(FIX, M, EVENT_ID0);
        WAIT_FLAG(FIX, M, EVENT_ID1);
    }

private:
    __gm__ uint8_t *__restrict__ tiling_para_gm;
    AscendC::GlobalTensor<int64_t> actualSeqLenGmTensorQ;
    AscendC::GlobalTensor<int64_t> actualSeqLenGmTensorKv;
    AscendC::GlobalTensor<int32_t> blockTableGmTensor;

    AscendC::GlobalTensor<Q_T> q_gm_tensor;
    AscendC::GlobalTensor<KV_T> dequantKeyWsPing_;
    AscendC::GlobalTensor<KV_T> dequantKeyWsPong_;
    AscendC::GlobalTensor<KV_T> dequantValueWsPing_;
    AscendC::GlobalTensor<KV_T> dequantValueWsPong_;

    AscendC::GlobalTensor<S_DATA_TYPE> s_gm_tensor;
    AscendC::GlobalTensor<Q_T> p_gm_tensor;
    AscendC::GlobalTensor<float> o_tmp_gm_tensor;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;

    AscendC::LocalTensor<Q_T> l1q_buf_addr_tensor;
    AscendC::LocalTensor<KV_T> l1k_buf_addr_tensor;
    AscendC::LocalTensor<Q_T> l1p_buf_addr_tensor;
    AscendC::LocalTensor<KV_T> l1v_buf_addr_tensor;

    AscendC::LocalTensor<Q_T> l0a_buf_tensor;
    AscendC::LocalTensor<KV_T> l0b_buf_tensor;
    AscendC::LocalTensor<float> l0c_buf_tensor;

    uint32_t batchSize_;
    uint32_t qHeadNum_;
    uint32_t embedding_size;
    uint32_t kvHeadNum_;
    uint32_t maxNumBlocksPerQuery;
    uint32_t kvBlockSize_;
    uint32_t totalBlockNumQ_;
    uint32_t groupNum;
    uint32_t stride_kv;
    uint32_t stride_qo;
    uint32_t headDim_;
    uint32_t round_k;
    uint32_t is_triu_mask = 0;
    uint32_t tiling_head_size;
    uint32_t tiling_para_size;
    uint32_t prefill_batch_size_;
    uint32_t decoder_batch_size_;
    uint32_t mask_type;
    uint32_t max_kv;

    uint32_t numBlocks_;
    uint32_t blockIdx_;
    uint32_t seqStepQ_;
    uint32_t seqStepKv_;
};

template <typename IFAT, TilingKeyType tilingKeyType = TilingKeyType::TILING_INT8_VEC_QUANT>
class PagedAttentionParallelAiv {
public:
    using IN_KV_TYPE = typename IFAT::kvType;
    using DEQUANT_KV_TYPE = typename IFAT::queryType;
    using S_DATA_TYPE = half;
    using P_DATA_TYPE = typename IFAT::outputType;
    using MASK_DATA_TYPE = half;
    using TILING_T = typename IFAT::TilingType;

    __aicore__ inline PagedAttentionParallelAiv(uint32_t prefill_batch_size, uint32_t decoder_batch_size) {
        prefill_batch_size_ = prefill_batch_size;
        decoder_batch_size_ = decoder_batch_size;
    }

    __aicore__ __attribute__((always_inline)) inline void InitQuant(
        __gm__ uint8_t *__restrict__ antiquant_scale,
        __gm__ uint8_t *__restrict__ antiquant_offset
    )
    {
        kvAntiquantBiasFlag_ = (antiquant_offset != nullptr);

        // 输入，在反量化的时候用到
        antiquantScaleGmTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(antiquant_scale));
        antiquantOffsetGmTensor_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(antiquant_offset));
    }

    __aicore__ inline void DequantKV(GlobalTensor<DEQUANT_KV_TYPE> dstGmTensor,       // [qk_n, sub_m, embedding_size]
                                     GlobalTensor<int8_t> srcGmTensor,         // [num_blocks, kvBlockSize_, hidden_size]
                                     GlobalTensor<float> deq_offset, // [hidden_size,]
                                     GlobalTensor<float> deq_scale,    // [hidden_size,]
                                     const uint32_t hidden_size, const uint32_t batch_idx, const uint32_t n_idx,
                                     const uint32_t kv_seq_len, const uint32_t sub_m, const uint32_t hiddenSizeOffsetKv,
                                     const uint32_t real_n_loop, const uint32_t sub_n_loop, const uint32_t start_kv,
                                     const uint32_t bias_flag)
    {
        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        uint32_t start_seq = 0;
        uint32_t stepNumPerBlock = kvBlockSize_ / seqStepKv_;
        PIPE_BARRIER(ALL);

        // [qk_n, sub_m, head_size]
        SET_FLAG(V, MTE2, EVENT_ID5);
        SET_FLAG(MTE3, V, EVENT_ID5);
        SET_FLAG(V, MTE2, EVENT_ID6);
        SET_FLAG(MTE3, V, EVENT_ID6);
        uint32_t sub_hiddensize = sub_m * embedding_size;
        uint32_t sub_hidden_d32 = RoundUp<CONST_32>(sub_hiddensize);
        uint32_t sub_hidden_d64 = RoundUp<CONST_64>(sub_hiddensize);
        uint32_t num_deq_kv = FLOAT_VECTOR_SIZE_D * UB_FLOAT_LINE_SIZE_D;
        uint32_t kv_seq_step = num_deq_kv / sub_hidden_d32;
        for (uint32_t ni = 0; ni < sub_n_loop; ++ni) {
            uint32_t actual_idx = n_idx * sub_n_loop + ni;
            uint32_t kvBlockIdx = actual_idx / stepNumPerBlock;
            uint32_t stepIdxInKvBlock = actual_idx % stepNumPerBlock;
            if (actual_idx >= real_n_loop) {
                break;
            }
            uint32_t sub_qk_n = (actual_idx != real_n_loop - 1) ? seqStepKv_ : kv_seq_len - actual_idx * seqStepKv_;
            uint32_t page_idx = blockTableGmTensor.GetValue(batch_idx * maxNumBlocksPerQuery + start_kv / kvBlockSize_ + kvBlockIdx);

            uint32_t dequant_ping_pang = 0;
            // [sub_qk_n, sub_m, head_size]
            for (uint32_t si = 0; si < sub_qk_n; si += kv_seq_step) {
                // copy src from gm to ub
                uint32_t seq_len_frag = Min(sub_qk_n - si, kv_seq_step);
                WAIT_FLAG(V, MTE2, EVENT_ID5 + dequant_ping_pang);
                uint32_t srcGmOffset = (page_idx * kvBlockSize_ + stepIdxInKvBlock * seqStepKv_ + si) * hidden_size + hiddenSizeOffsetKv;
                gm_to_ub_align<ArchType::ASCEND_V220, int8_t>(kvUbTensorInt8_[dequant_ping_pang * num_deq_kv],
                                                              srcGmTensor[srcGmOffset],
                                                              0,                              // sid
                                                              seq_len_frag,                   // nBurst
                                                              sub_hiddensize,                 // lenBurst
                                                              0,                              // leftPaddingNum
                                                              0,                              // rightPaddingNum
                                                              (hidden_size - sub_hiddensize), // srcGap
                                                              0                               // dstGap
                    );

                if (si == 0 && ni == 0) {
                    if (bias_flag) {
                        // copy deq_offset from gm to ub
                        gm_to_ub_align<ArchType::ASCEND_V220, float>(dequantOffsetUbTensorFp32_,
                                                                       deq_offset,
                                                                       0,                  // sid
                                                                       1,                  // nBurst
                                                                       sub_hiddensize * FLOAT_BYTES, // lenBurst
                                                                       0,                  // leftPaddingNum
                                                                       0,                  // rightPaddingNum
                                                                       0,                  // srcGap
                                                                       0                   // dstGap
                        );
                    }
                    gm_to_ub_align<ArchType::ASCEND_V220, float>(dequantScaleUbTensor_, deq_scale,
                                                                 0,                  // sid
                                                                 1,                  // nBurst
                                                                 sub_hiddensize * FLOAT_BYTES, // lenBurst
                                                                 0,                  // leftPaddingNum
                                                                 0,                  // rightPaddingNum
                                                                 0,                  // srcGap
                                                                 0                   // dstGap
                    );
                }
                SET_FLAG(MTE2, V, EVENT_ID5 + dequant_ping_pang);
                WAIT_FLAG(MTE2, V, EVENT_ID5 + dequant_ping_pang);

                // cast src(int8) -> src(fp16) -> src(int32)
                uint32_t numel_kv = seq_len_frag * sub_hidden_d32;
                uint32_t count = numel_kv / MAX_NUMEL_INST_B16;

                WAIT_FLAG(MTE3, V, EVENT_ID5 + dequant_ping_pang);
                for (uint32_t i = 0; i < count; ++i) {
                    conv_v<ArchType::ASCEND_V220, int8_t, half>(
                        kvUbTensorHalf_.template ReinterpretCast<half>()[dequant_ping_pang * num_deq_kv + i * MAX_NUMEL_INST_B16],       // dst
                        kvUbTensorInt8_[dequant_ping_pang * num_deq_kv + i * MAX_NUMEL_INST_B16], // src
                        VEC_REPEAT_NUM_255,                                                                  // repeat
                        1,                                                                    // dstBlockStride
                        1,                                                                    // srcBlockStride
                        BLOCK_NUM_IN_REPEAT,                                                                    // dstRepeatStride
                        BLOCK_NUM_IN_REPEAT_HALF                                                                     // srcRepeatStride
                    );
                }
                conv_v<ArchType::ASCEND_V220, int8_t, half>(
                    kvUbTensorHalf_.template ReinterpretCast<half>()[dequant_ping_pang * num_deq_kv + count * MAX_NUMEL_INST_B16],     // dst
                    kvUbTensorInt8_[dequant_ping_pang * num_deq_kv + count * MAX_NUMEL_INST_B16], // src
                    CeilDiv<uint32_t>((numel_kv - count * MAX_NUMEL_INST_B16), HALF_VECTOR_SIZE),// repeat
                    1,                                                                        // dstBlockStride
                    1,                                                                        // srcBlockStride
                    BLOCK_NUM_IN_REPEAT,                                                                        // dstRepeatStride
                    BLOCK_NUM_IN_REPEAT_HALF                                                                         // srcRepeatStride
                );
                SET_FLAG(V, MTE2, EVENT_ID5 + dequant_ping_pang);
                // cast src(fp16) -> src(float)
                PIPE_BARRIER(V);
                count = numel_kv / MAX_NUMEL_INST_B32;
                for (uint32_t i = 0; i < count; ++i) {
                    conv_v<ArchType::ASCEND_V220, half, float>(
                        kvUbTensorFp32_[dequant_ping_pang * num_deq_kv + i * MAX_NUMEL_INST_B32], // dst
                        kvUbTensorHalf_.template ReinterpretCast<half>()[dequant_ping_pang * num_deq_kv + i * MAX_NUMEL_INST_B32], // src
                        VEC_REPEAT_NUM_255,                                   // repeat
                        1,                                     // dstBlockStride
                        1,                                     // srcBlockStride
                        BLOCK_NUM_IN_REPEAT,                                     // dstRepeatStride
                        BLOCK_NUM_IN_REPEAT_HALF                                      // srcRepeatStride
                    );
                }
                conv_v<ArchType::ASCEND_V220, half, float>(
                    kvUbTensorFp32_[dequant_ping_pang * num_deq_kv + count * MAX_NUMEL_INST_B32], // dst
                    kvUbTensorHalf_.template ReinterpretCast<half>()[dequant_ping_pang * num_deq_kv + count * MAX_NUMEL_INST_B32],                       // src
                    CeilDiv<uint32_t>((numel_kv - count * MAX_NUMEL_INST_B32), FLOAT_VECTOR_SIZE_D),// repeat
                    1,                                                           // dstBlockStride
                    1,                                                           // srcBlockStride
                    BLOCK_NUM_IN_REPEAT,                                                           // dstRepeatStride
                    BLOCK_NUM_IN_REPEAT_HALF                                                            // srcRepeatStride
                );
                if (bias_flag) {
                    // src(float) <- src(float) + offset(float)
                    PIPE_BARRIER(V);
                    count = sub_hiddensize / FLOAT_VECTOR_SIZE_D;
                    for (uint32_t i = 0; i < count; ++i) {
                        add_v<ArchType::ASCEND_V220, float>(
                            kvUbTensorFp32_[dequant_ping_pang * num_deq_kv + i * FLOAT_VECTOR_SIZE_D], // dst
                            kvUbTensorFp32_[dequant_ping_pang * num_deq_kv + i * FLOAT_VECTOR_SIZE_D], // src0
                            dequantOffsetUbTensorFp32_[i * FLOAT_VECTOR_SIZE_D],                                // src1
                            seq_len_frag,                                                        // repeat
                            BLK_STRIDE_ONE,                                                                   // dstBlockStride
                            BLK_STRIDE_ONE,                                                                   // src0BlockStride
                            BLK_STRIDE_ONE,                                                                   // src1BlockStride
                            sub_hidden_d32 / BLOCK_NUM_IN_REPEAT,                                                  // dstRepeatStride
                            sub_hidden_d32 / BLOCK_NUM_IN_REPEAT,                                                  // src0RepeatStride
                            0                                                                    // src1RepeatStride
                        );
                    }
                    if (sub_hiddensize % FLOAT_VECTOR_SIZE_D > 0) {
                        set_mask_d(sub_hiddensize % FLOAT_VECTOR_SIZE_D);
                        add_v<ArchType::ASCEND_V220, float>(
                            kvUbTensorFp32_[dequant_ping_pang * num_deq_kv + count * FLOAT_VECTOR_SIZE_D], // dst
                            kvUbTensorFp32_[dequant_ping_pang * num_deq_kv + count * FLOAT_VECTOR_SIZE_D], // src0
                            dequantOffsetUbTensorFp32_[count * FLOAT_VECTOR_SIZE_D],                                // src1
                            seq_len_frag,                                                            // repeat
                            1,                                                                       // dstBlockStride
                            1,                                                                       // src0BlockStride
                            1,                                                                       // src1BlockStride
                            sub_hidden_d32 / BLOCK_NUM_IN_REPEAT,                                                      // dstRepeatStride
                            sub_hidden_d32 / BLOCK_NUM_IN_REPEAT,                                                      // src0RepeatStride
                            0                                                                        // src1RepeatStride
                        );
                        SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                    }
                }
                // src(float) <- src(float) * scale(float)
                PIPE_BARRIER(V);
                count = sub_hiddensize / FLOAT_VECTOR_SIZE_D;
                for (uint32_t i = 0; i < count; ++i) {
                    mul_v<ArchType::ASCEND_V220, float>(
                        kvUbTensorFp32_[dequant_ping_pang * num_deq_kv + FLOAT_VECTOR_SIZE_D * i], // dst
                        kvUbTensorFp32_[dequant_ping_pang * num_deq_kv + FLOAT_VECTOR_SIZE_D * i], // src0
                        dequantScaleUbTensor_[FLOAT_VECTOR_SIZE_D * i],                                    // src1
                        seq_len_frag,                                                        // repeat
                        1,                                                                   // dstBlockStride
                        1,                                                                   // src0BlockStride
                        1,                                                                   // src1BlockStride
                        sub_hidden_d32 / BLOCK_NUM_IN_REPEAT,                                // dstRepeatStride
                        sub_hidden_d32 / BLOCK_NUM_IN_REPEAT,                                // src0RepeatStride
                        0                                                                    // src1RepeatStride
                    );
                }
                // 非对齐场景处理：偏移的数据量按照对齐偏移sub_hidden_d32
                if (sub_hiddensize % FLOAT_VECTOR_SIZE_D > 0) {
                    set_mask_d(sub_hiddensize % FLOAT_VECTOR_SIZE_D);
                    mul_v<ArchType::ASCEND_V220, float>(
                        kvUbTensorFp32_[dequant_ping_pang * num_deq_kv + count * FLOAT_VECTOR_SIZE_D], // dst
                        kvUbTensorFp32_[dequant_ping_pang * num_deq_kv + count * FLOAT_VECTOR_SIZE_D], // src0
                        dequantScaleUbTensor_[count * FLOAT_VECTOR_SIZE_D],                                    // src1
                        seq_len_frag,                                                            // repeat
                        1,                                                                       // dstBlockStride
                        1,                                                                       // src0BlockStride
                        1,                                                                       // src1BlockStride
                        sub_hidden_d32 / BLOCK_NUM_IN_REPEAT,                                                      // dstRepeatStride
                        sub_hidden_d32 / BLOCK_NUM_IN_REPEAT,                                                      // src0RepeatStride
                        0                                                                        // src1RepeatStride
                    );
                }
                // cast src(float) -> src(half)
                count = numel_kv / MAX_NUMEL_INST_B32;
                SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                PIPE_BARRIER(V);
                for (uint32_t i = 0; i < count; ++i) {
                    conv_v<ArchType::ASCEND_V220, float, DEQUANT_KV_TYPE>(
                        kvUbTensorDequant_[dequant_ping_pang * num_deq_kv + i * MAX_NUMEL_INST_B32], // dst
                        kvUbTensorFp32_[dequant_ping_pang * num_deq_kv + i * MAX_NUMEL_INST_B32], // src
                        VEC_REPEAT_NUM_255,                                                                  // repeat
                        1,                                                                    // dstBlockStride
                        1,                                                                    // srcBlockStride
                        BLOCK_NUM_IN_REPEAT_HALF,                                                                    // dstRepeatStride
                        BLOCK_NUM_IN_REPEAT                                                                     // srcRepeatStride
                    );
                }
                conv_v<ArchType::ASCEND_V220, float, DEQUANT_KV_TYPE>(
                    kvUbTensorDequant_[dequant_ping_pang * num_deq_kv + count * MAX_NUMEL_INST_B32], // dst
                    kvUbTensorFp32_[dequant_ping_pang * num_deq_kv + count * MAX_NUMEL_INST_B32], // src
                    CeilDiv<uint32_t>((numel_kv - count * MAX_NUMEL_INST_B32), FLOAT_VECTOR_SIZE_D),// repeat
                    1,                                                                        // dstBlockStride
                    1,                                                                        // srcBlockStride
                    BLOCK_NUM_IN_REPEAT_HALF,                                                                        // dstRepeatStride
                    BLOCK_NUM_IN_REPEAT                                                                         // srcRepeatStride
                );
                SET_FLAG(V, MTE3, EVENT_ID0 + dequant_ping_pang);
                WAIT_FLAG(V, MTE3, EVENT_ID0 + dequant_ping_pang);

                // 非对齐场景处理：非对齐headsize需要依照情况手动偏移dummydata
                uint32_t align_size = (sub_m * embedding_size % CONST_32);
                uint32_t padd_gap = (align_size > 0) ? ((UB_ALIGN_BYTE - align_size) >= CONST_16 ? 1 : 0) : 0;
                ub_to_gm_align<ArchType::ASCEND_V220, DEQUANT_KV_TYPE>(dstGmTensor[(start_seq + si) * sub_hiddensize],
                                                            kvUbTensorDequant_[dequant_ping_pang * num_deq_kv],
                                                                0,                          // sid
                                                                seq_len_frag,               // nBurst
                                                                sub_m * embedding_size * HALF_BYTES, // lenBurst
                                                                0,                          // leftPaddingNum
                                                                0,                          // rightPaddingNum
                                                                padd_gap,                   // srcGap
                                                                0                           // dstGap
                );

                SET_FLAG(MTE3, V, EVENT_ID5 + dequant_ping_pang);
                dequant_ping_pang = 1 - dequant_ping_pang;
            }
            start_seq += sub_qk_n;
        }
        WAIT_FLAG(MTE3, V, EVENT_ID5);
        WAIT_FLAG(V, MTE2, EVENT_ID5);
        WAIT_FLAG(MTE3, V, EVENT_ID6);
        WAIT_FLAG(V, MTE2, EVENT_ID6);
    }

    __aicore__ inline void SetArgs(
        __gm__ uint8_t *__restrict__ k_in_gm,
        __gm__ uint8_t *__restrict__ v_in_gm,
        __gm__ uint8_t *__restrict__ seq_len_q_gm,
        __gm__ uint8_t *__restrict__ seq_len_kv_gm,
        __gm__ uint8_t *__restrict__ block_tables_in_gm,
        __gm__ uint8_t *__restrict__ mask_in_gm,
        __gm__ uint8_t *__restrict__ o_out_gm,
        __gm__ uint8_t *__restrict__ workspace_gm,
        const typename IFAT::TilingType *__restrict tiling)
    {
        mask_gm = mask_in_gm;

        ListTensorDesc keyListTensorDesc((__gm__ void *)k_in_gm);
        ListTensorDesc valueListTensorDesc((__gm__ void *)v_in_gm);
        key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
        value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);

        k_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ IN_KV_TYPE *>(key_));
        v_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ IN_KV_TYPE *>(value_));

        mask_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ MASK_DATA_TYPE *>(mask_in_gm));
        o_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ P_DATA_TYPE *>(o_out_gm));
        
        actualSeqLenGmTensorQ.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(seq_len_q_gm));
        actualSeqLenGmTensorKv.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(seq_len_kv_gm));
        blockTableGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(block_tables_in_gm));

        const uint32_t addr_kv8 = 0;
        const uint32_t addr_kv16 = addr_kv8 + CONST_32 * CONST_4 * CONST_128 * sizeof(int8_t);
        const uint32_t addr_kv32 = addr_kv16 + CONST_32 * CONST_4 * CONST_128 * sizeof(half);
        const uint32_t addr_offset = addr_kv32 + CONST_32 * CONST_4 * CONST_128 * FLOAT_BYTES;
        const uint32_t addr_scale = addr_offset + CONST_8 * CONST_128 * FLOAT_BYTES;
        const uint32_t addr_offset32 = addr_scale + CONST_8 * CONST_128 * FLOAT_BYTES;

        const uint32_t ls_ubuf_offset = 0;
        const uint32_t lp_ubuf_offset = 0;
        const uint32_t ls32_ubuf_offset = CONST_TWO * PRE_UB_UINT8_BLOCK_SIZE;
        const uint32_t mask_ubuf_offset = CONST_4 * PRE_UB_UINT8_BLOCK_SIZE;
        const uint32_t lo_ubuf_offset = CONST_6 * PRE_UB_UINT8_BLOCK_SIZE;
        const uint32_t lm_ubuf_offset = CONST_8 * PRE_UB_UINT8_BLOCK_SIZE;
        const uint32_t hm_ubuf_offset = CONST_8 * PRE_UB_UINT8_BLOCK_SIZE + 1 * UB_UINT8_LINE_SIZE_D;
        const uint32_t gm_ubuf_offset = CONST_8 * PRE_UB_UINT8_BLOCK_SIZE + CONST_TWO * UB_UINT8_LINE_SIZE_D;
        const uint32_t dm_ubuf_offset = CONST_8 * PRE_UB_UINT8_BLOCK_SIZE + CONST_4 * UB_UINT8_LINE_SIZE_D;
        const uint32_t ll_ubuf_offset = CONST_8 * PRE_UB_UINT8_BLOCK_SIZE + CONST_8 * UB_UINT8_LINE_SIZE_D;
        const uint32_t gl_ubuf_offset = CONST_8 * PRE_UB_UINT8_BLOCK_SIZE + CONST_16 * UB_UINT8_LINE_SIZE_D;
        const uint32_t tv_ubuf_offset = CONST_8 * PRE_UB_UINT8_BLOCK_SIZE + CONST_17 * UB_UINT8_LINE_SIZE_D;
        const uint32_t go_ubuf_offset = CONST_9 * PRE_UB_UINT8_BLOCK_SIZE;
        const uint32_t mask16_ubuf_offset = CONST_11 * PRE_UB_UINT8_BLOCK_SIZE;

        kvUbTensorInt8_ = buf.GetBuffer<BufferType::ASCEND_UB, int8_t>(addr_kv8);
        kvUbTensorHalf_ = buf.GetBuffer<BufferType::ASCEND_UB, half>(addr_kv16);
        // 和kvUbTensorHalf_是相同的位置，用来转回bf16/fp16
        kvUbTensorDequant_ = buf.GetBuffer<BufferType::ASCEND_UB, DEQUANT_KV_TYPE>(addr_kv16);
        kvUbTensorFp32_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(addr_kv32);
        dequantOffsetUbTensor_ = buf.GetBuffer<BufferType::ASCEND_UB, int32_t>(addr_offset);
        dequantScaleUbTensor_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(addr_scale);
        dequantOffsetUbTensorFp32_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(addr_offset32);

        ls_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, S_DATA_TYPE>(ls_ubuf_offset);
        lp_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, P_DATA_TYPE>(lp_ubuf_offset);
        ls32_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(ls32_ubuf_offset);
        mask_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, S_DATA_TYPE>(mask_ubuf_offset);
        lo_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(lo_ubuf_offset);
        lm_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, S_DATA_TYPE>(lm_ubuf_offset);
        hm_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, S_DATA_TYPE>(hm_ubuf_offset);
        gm_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, S_DATA_TYPE>(gm_ubuf_offset);
        dm_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, S_DATA_TYPE>(dm_ubuf_offset);
        ll_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(ll_ubuf_offset);
        gl_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(gl_ubuf_offset);
        tv_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(tv_ubuf_offset);
        go_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(go_ubuf_offset);
        mask16_ubuf_tensor = buf.GetBuffer<BufferType::ASCEND_UB, MASK_DATA_TYPE>(mask16_ubuf_offset);

        batchSize_ = tiling->tilingBase.batchSize;
        qHeadNum_ = tiling->tilingBase.qHeadNum;
        embedding_size = tiling->tilingBase.headSize;
        kvHeadNum_ = tiling->tilingBase.kvHeadNum;
        kvBlockSize_ = tiling->tilingBase.blockSize;
        seqStepQ_ = tiling->tilingPerCore.seqStepQ;
        seqStepKv_ = tiling->tilingPerCore.seqStepKv;
        tor = tiling->tilingBase.scaleValue;
        totalBlockNumQ_ = tiling->tilingPerCore.totalQBlockNum;
        batch_stride = tiling->tilingPerCore.maskBatchStride;
        head_stride = tiling->tilingPerCore.maskHeadStride;
        mask_type = tiling->tilingBase.attenMaskFlag;
        maxNumBlocksPerQuery = tiling->tilingBase.maxBlockNumPerBatch;
        groupNum = qHeadNum_ / kvHeadNum_;
        stride_qo = qHeadNum_ * embedding_size;

        headDim_ = embedding_size;
        round_k = (headDim_ + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16 * BLOCK_SIZE_16;

        numBlocks_ = GetBlockNum();
        // AIV的blockIdx会默认*2, 除掉后和AIC对应上
        blockIdx_ = GetBlockIdx() / AIC_AIV_RATIO;
        subBlockIdx_ = GetSubBlockIdx();

        uint32_t dequantKeyPingOffset = 0;
        // 每一次只涉及一个qHead和kvHead, 所以*1
        uint32_t dequantKeyPongOffset = dequantKeyPingOffset + numBlocks_ * BLOCKSIZE_CALC_256 * 1 * embedding_size * sizeof(DEQUANT_KV_TYPE);
        uint32_t dequantValuePingOffset = dequantKeyPongOffset + numBlocks_ * BLOCKSIZE_CALC_256 * 1 * embedding_size * sizeof(DEQUANT_KV_TYPE);
        uint32_t dequantValuePongOffset = dequantValuePingOffset + numBlocks_ * BLOCKSIZE_CALC_256 * 1 * embedding_size * sizeof(DEQUANT_KV_TYPE);
        uint32_t scoreWsOffset = dequantValuePongOffset + numBlocks_ * BLOCKSIZE_CALC_256 * 1 * embedding_size * sizeof(DEQUANT_KV_TYPE);
        uint32_t probWsOffset = scoreWsOffset + numBlocks_ * TMP_SIZE_64K * sizeof(S_DATA_TYPE);
        uint32_t outputTmpOffset = probWsOffset + numBlocks_ * TMP_SIZE_64K * sizeof(P_DATA_TYPE);

        dequantKeyWsPing_.SetGlobalBuffer(reinterpret_cast<__gm__ DEQUANT_KV_TYPE*>(workspace_gm + dequantKeyPingOffset));
        dequantKeyWsPong_.SetGlobalBuffer(reinterpret_cast<__gm__ DEQUANT_KV_TYPE*>(workspace_gm + dequantKeyPongOffset));
        dequantValueWsPing_.SetGlobalBuffer(reinterpret_cast<__gm__ DEQUANT_KV_TYPE*>(workspace_gm + dequantValuePingOffset));
        dequantValueWsPong_.SetGlobalBuffer(reinterpret_cast<__gm__ DEQUANT_KV_TYPE*>(workspace_gm + dequantValuePongOffset));

        s_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ S_DATA_TYPE *>(workspace_gm + scoreWsOffset));
        p_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ P_DATA_TYPE *>(workspace_gm + probWsOffset));
        o_tmp_gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_gm + outputTmpOffset));
    }

    __aicore__ inline void Run()
    {
        SET_FLAG(MTE3, MTE2, EVENT_ID0);
        SET_FLAG(MTE3, MTE2, EVENT_ID1);
        SET_FLAG(V, MTE2, EVENT_ID0);
        SET_FLAG(V, MTE2, EVENT_ID1);
        SET_FLAG(V, MTE2, EVENT_ID2);
        SET_FLAG(MTE3, V, EVENT_ID0);

        uint32_t go_flag_scalar = 1;

        uint32_t curBatchIdx = 0;
        // kvSeqLen相关数据
        uint32_t preAccumSeqLenKv = 0;
        uint32_t accumSeqLenKv = actualSeqLenGmTensorKv.GetValue(curBatchIdx);
        uint32_t kvSeqLenCurBatch = accumSeqLenKv - preAccumSeqLenKv;

        // seqLenQ相关数据
        uint32_t preAccumBlkNumQ = 0;
        uint32_t preAccumSeqLenQ = 0;
        uint32_t accumSeqLenQ = actualSeqLenGmTensorQ.GetValue(curBatchIdx);
        uint32_t qSeqLenCurBatch = accumSeqLenQ - preAccumSeqLenQ;
        uint32_t accumQueryBlkNum = preAccumBlkNumQ + (qSeqLenCurBatch + seqStepQ_ - 1) / seqStepQ_;
        uint32_t totalProcessNum = totalBlockNumQ_ * qHeadNum_;

        for (uint32_t process = 0; process < totalProcessNum; process++) {
            if (process >= accumQueryBlkNum * qHeadNum_) {
                bool doRunCircleFlag = true;
                while (doRunCircleFlag) {
                    curBatchIdx++;
                    // 记录上一次的累计数据
                    preAccumBlkNumQ = accumQueryBlkNum;
                    preAccumSeqLenQ = accumSeqLenQ;
                    preAccumSeqLenKv = accumSeqLenKv;

                    accumSeqLenKv = actualSeqLenGmTensorKv.GetValue(curBatchIdx);
                    kvSeqLenCurBatch = accumSeqLenKv - preAccumSeqLenKv;
                    accumSeqLenQ = actualSeqLenGmTensorQ.GetValue(curBatchIdx);
                    qSeqLenCurBatch = accumSeqLenQ - preAccumSeqLenQ;
                    accumQueryBlkNum += (qSeqLenCurBatch + seqStepQ_ - 1) / seqStepQ_;
                    if(qSeqLenCurBatch != 0) {
                        break;
                    }

                    if (curBatchIdx >= batchSize_ - 1) {
                        doRunCircleFlag = false;
                        break;
                    }
                }
            }

            uint32_t targetAicIdx = process % numBlocks_;
            if (is_triu_mask) {
                if ((process / numBlocks_) % TASK_ROUND_TWO == 1) {
                    targetAicIdx = numBlocks_ - process % numBlocks_ - 1;
                }
            }
            if (blockIdx_ != targetAicIdx) {
                continue;
            }

            uint64_t outputOffsetBase = preAccumSeqLenQ * qHeadNum_ * embedding_size;
            uint64_t mask_scalar = 0;

            uint32_t process_idx = process - preAccumBlkNumQ * qHeadNum_;
            uint32_t m_idx = process_idx / qHeadNum_;
            uint32_t head_idx = process_idx % qHeadNum_;

            uint32_t m_loop = (qSeqLenCurBatch + seqStepQ_ - 1) / seqStepQ_;
            uint32_t n_loop = (kvSeqLenCurBatch + seqStepKv_ - 1) / seqStepKv_;

            uint32_t qk_m = (m_idx == (m_loop - 1)) ? (qSeqLenCurBatch - m_idx * seqStepQ_) : seqStepQ_;
            uint32_t sub_m = (subBlockIdx_ == 1) ? (qk_m - qk_m / AIC_AIV_RATIO) : qk_m / AIC_AIV_RATIO;
            uint32_t sub_m_d128 = (sub_m + VECTOR_SIZE_D - 1) / VECTOR_SIZE_D;             // up aligned to 128
            uint32_t sub_m_d64 = (sub_m + FLOAT_VECTOR_SIZE_D - 1) / FLOAT_VECTOR_SIZE_D;  // up aligned to 64
            uint32_t round_sub_m = (sub_m + BLOCK_SIZE_D - 1) / BLOCK_SIZE_D * BLOCK_SIZE_D;

            // dequant kv的时候两个vec切分kvHead
            uint32_t kv_start_head = head_idx / groupNum;
            uint32_t kv_end_head = kv_start_head + 1;
            uint32_t cur_kvhead_num = kv_end_head - kv_start_head;
            uint32_t sub_m_kv = (subBlockIdx_ == 1) ? (cur_kvhead_num - cur_kvhead_num / AIC_AIV_RATIO) : cur_kvhead_num / AIC_AIV_RATIO;
            uint32_t hiddenSizeOffsetKv = kv_start_head * embedding_size;
            uint32_t hiddenSizeOffsetWithBiasKv = kvAntiquantBiasFlag_ ? hiddenSizeOffsetKv : 0;
            uint32_t dequantWsOffset = blockIdx_ * BLOCKSIZE_CALC_256 * 1 * embedding_size;
            uint32_t sub_n_loop = 1;  // pp_n_scalar / kvBlockSize_;
            uint32_t start_kv = 0;
            uint32_t pingpong_flag_dequant = 0;

            uint64_t qk_index = 0;
            /******** pre_load *******/
            uint32_t qk_n = (qk_index == (n_loop - 1)) ? kvSeqLenCurBatch : seqStepKv_;
            uint32_t qk_round_n = (qk_n + BLOCK_SIZE_D - 1) / BLOCK_SIZE_D * BLOCK_SIZE_D;

            uint32_t pingpong_flag = 0;
            uint32_t offset = pingpong_flag * UB_HALF_BUF_SIZE_D;
            uint64_t mask_offset = (curBatchIdx * batch_stride + head_idx * head_stride) * max_context_len + m_idx * seqStepQ_ * max_context_len;
            mask_offset += mask_scalar;

            uint64_t o_offset = outputOffsetBase + head_idx * embedding_size + m_idx * seqStepQ_ * stride_qo;
            uint32_t n_end = n_loop;
            if (is_triu_mask) {
                uint32_t triu_seq = mask_type == 3 ? max_kv - qSeqLenCurBatch : kvSeqLenCurBatch - qSeqLenCurBatch;
                uint32_t n_offset = ((m_idx + 1) * seqStepQ_ + triu_seq + seqStepKv_ - 1) / seqStepKv_;
                n_end = n_offset > n_loop ? n_loop : n_offset;
            }
            uint32_t KV_INC = 1;
            uint32_t vect_mod = AIC_AIV_RATIO * KV_INC;
            uint32_t long_seq = 0;
            for (uint32_t n_idx = 0; n_idx < n_end + KV_INC; n_idx += KV_INC) {
                if (n_idx < n_end) {
                    if constexpr (tilingKeyType == TilingKeyType::TILING_INT8_VEC_QUANT || tilingKeyType == TilingKeyType::TILING_INT8_VEC_QUANTBF16) {                        
                        for (uint32_t split_idx = 0; split_idx < KV_INC && n_idx + split_idx < n_end; split_idx++) {
                            pingpong_flag_dequant = (n_idx + split_idx) % DOUBLE_BUFFER_NUM;
                            uint32_t dequantKeySyncFlag = VEC_DEQ_K0_READY + pingpong_flag_dequant;
                            AscendC::GlobalTensor<DEQUANT_KV_TYPE> dequantKeyGm = pingpong_flag_dequant ? dequantKeyWsPong_ : dequantKeyWsPing_;
                            if (sub_m_kv > 0) {
                                DequantKV(dequantKeyGm[dequantWsOffset],                                        // dst
                                        k_gm_tensor,                                                            // src
                                        antiquantOffsetGmTensor_[hiddenSizeOffsetWithBiasKv],                   // deq_offset
                                        antiquantScaleGmTensor_[hiddenSizeOffsetKv],                            // deq_scale
                                        kvHeadNum_ * embedding_size,                                            // hidden_size
                                        curBatchIdx,                                                              // batch_idx
                                        n_idx + split_idx,                                                      // n_idx
                                        kvSeqLenCurBatch,                                                              // kv_seq_len
                                        sub_m_kv,                                                               // sub_m
                                        hiddenSizeOffsetKv,                                                     // hiddenSizeOffsetKv
                                        n_loop,                                                                 // real_n_loop
                                        sub_n_loop,                                                             // sub_n_loop
                                        start_kv,                                                               // start_kv
                                        kvAntiquantBiasFlag_);
                            }
                            AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_MTE3>(dequantKeySyncFlag);
                        }

                        for (uint32_t split_idx = 0; split_idx < KV_INC && n_idx + split_idx < n_end; split_idx++) {
                            pingpong_flag_dequant = (n_idx + split_idx) % DOUBLE_BUFFER_NUM;
                            uint32_t dequantValueSyncFlag = VEC_DEQ_V0_READY + pingpong_flag_dequant;
                            AscendC::GlobalTensor<DEQUANT_KV_TYPE> dequantValueGm = pingpong_flag_dequant ? dequantValueWsPong_ : dequantValueWsPing_;
                            if (sub_m_kv > 0) {
                                DequantKV(dequantValueGm[dequantWsOffset], // dst
                                        v_gm_tensor,                                                           // src
                                        antiquantOffsetGmTensor_[hiddenSizeOffsetWithBiasKv],                                    // deq_offset
                                        antiquantScaleGmTensor_[hiddenSizeOffsetKv],                                            // deq_scale
                                        kvHeadNum_ * embedding_size,                                     // hidden_size
                                        curBatchIdx,                                                         // batch_idx
                                        split_idx + n_idx,                                                             // n_idx
                                        kvSeqLenCurBatch,                                                     // kv_seq_len
                                        sub_m_kv,                                                             // sub_m
                                        hiddenSizeOffsetKv,                                                      // hiddenSizeOffsetKv
                                        n_loop,                                                       // real_n_loop
                                        sub_n_loop,                                                        // sub_n_loop
                                        start_kv,                                                          // start_kv
                                        kvAntiquantBiasFlag_);
                            }
                            AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_MTE3>(dequantValueSyncFlag);
                        }
                    }
                    if (sub_m > 0 && mask_gm != nullptr) {
                        uint32_t qk_n_temp = qk_n;
                        for (uint32_t split_idx = 0; split_idx < KV_INC && n_idx + split_idx < n_end; split_idx++) {
                            WAIT_FLAG(V, MTE2, split_idx * CONST_TWO);
                            if (n_idx + split_idx == (n_loop - 1)) {
                                qk_n_temp = (kvSeqLenCurBatch - (n_idx + split_idx) * seqStepKv_);
                            }
                            if (long_seq == 0) {
                                gm_to_ub_align<ArchType::ASCEND_V220, half>(
                                    mask_ubuf_tensor[split_idx * UB_HALF_BUF_SIZE_D],
                                    mask_gm_tensor[mask_offset + (uint64_t)subBlockIdx_ * qk_m / AIC_AIV_RATIO * max_context_len],
                                    0,                                 // sid
                                    sub_m,                             // nBurst
                                    qk_n_temp * CONST_TWO,                     // lenBurst
                                    0,                                 // leftPaddingNum
                                    0,                                 // rightPaddingNum
                                    (max_context_len - qk_n_temp) * CONST_TWO, // srcGap
                                    0                                  // dstGap
                                );
                                mask_offset += seqStepKv_;
                            }
                        }
                    }
                    if ASCEND_IS_AIV {
                        AscendC::CrossCoreWaitFlag(QK_READY_FLAG);
                    }
                    
                    for (uint32_t split_idx = 0; split_idx < KV_INC && n_idx + split_idx < n_end; split_idx++) {
                        pingpong_flag = (n_idx + split_idx) % DOUBLE_BUFFER_NUM;
                        offset = pingpong_flag * UB_HALF_BUF_SIZE_D;
                        if (n_idx + split_idx == (n_loop - 1)) {
                            qk_n = (kvSeqLenCurBatch - (n_idx + split_idx) * seqStepKv_);
                            qk_round_n = (qk_n + BLOCK_SIZE_D - 1) / BLOCK_SIZE_D * BLOCK_SIZE_D;
                        }
                        if (sub_m > 0) {
                            WAIT_FLAG(MTE3, MTE2, pingpong_flag);
                            // input QK
                            gm_to_ub<ArchType::ASCEND_V220, half>(
                                ls_ubuf_tensor[offset],
                                s_gm_tensor[(uint64_t)blockIdx_ * TMP_SIZE_64K + (n_idx + split_idx) % vect_mod * TMP_SIZE_64K / vect_mod +
                                    (uint64_t)subBlockIdx_ * qk_m / AIC_AIV_RATIO * qk_round_n],
                                0,                                // sid
                                1,                                // nBurst
                                sub_m * qk_round_n / BLOCK_SIZE_D,  // lenBurst
                                0,                                // srcGap
                                0                                 // dstGap
                            );
                            SET_FLAG(MTE2, V, EVENT_ID0);
                            WAIT_FLAG(MTE2, V, EVENT_ID0);
                            // *** ls = tor * ls
                            for (uint32_t vadd_idx = 0; vadd_idx < qk_n / VECTOR_SIZE_D; ++vadd_idx) {
                                muls_v<ArchType::ASCEND_V220, half>(ls_ubuf_tensor[offset + vadd_idx * VECTOR_SIZE_D],
                                    ls_ubuf_tensor[offset + vadd_idx * VECTOR_SIZE_D],
                                    tor,
                                    sub_m,                          // repeat
                                    1,                              // dstBlockStride
                                    1,                              // srcBlockStride
                                    qk_round_n / BLOCK_SIZE_D,  // dstRepeatStride
                                    qk_round_n / BLOCK_SIZE_D  // srcRepeatStride
                                );
                            }
                            if (qk_n % VECTOR_SIZE_D > 0) {
                                set_mask_d(qk_n % VECTOR_SIZE_D);
                                muls_v<ArchType::ASCEND_V220, half>(ls_ubuf_tensor[offset + qk_n / VECTOR_SIZE_D * VECTOR_SIZE_D],
                                    ls_ubuf_tensor[offset + qk_n / VECTOR_SIZE_D * VECTOR_SIZE_D],
                                    tor,
                                    sub_m,                          // repeat
                                    1,                              // dstBlockStride
                                    1,                              // srcBlockStride
                                    qk_round_n / BLOCK_SIZE_D,  // dstRepeatStride
                                    qk_round_n / BLOCK_SIZE_D  // srcRepeatStride
                                );
                                SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                            }
                            PIPE_BARRIER(V);

                            // *** ls = ls + mask
                            if (mask_gm != nullptr) {
                                if (long_seq == 0) {
                                    add_v<ArchType::ASCEND_V220, half>(ls_ubuf_tensor[offset], ls_ubuf_tensor[offset],
                                        mask_ubuf_tensor[split_idx * UB_HALF_BUF_SIZE_D],
                                        (sub_m * qk_round_n + VECTOR_SIZE_D - 1) / VECTOR_SIZE_D, // repeat
                                        1,                                                    // dstBlockStride
                                        1,                                                    // src0BlockStride
                                        1,                                                    // src1BlockStride
                                        BLOCK_NUM_IN_REPEAT,                                                    // dstRepeatStride
                                        BLOCK_NUM_IN_REPEAT,                                                    // src0RepeatStride
                                        BLOCK_NUM_IN_REPEAT                                                     // src1RepeatStride
                                    );
                                    PIPE_BARRIER(V);
                                }
                                SET_FLAG(V, MTE2, split_idx * CONST_TWO);
                            }
                            // *** lm = rowmax(ls)
                            if (qk_n <= VECTOR_SIZE_D) {
                                set_mask_d(qk_n);
                                cmax_v<ArchType::ASCEND_V220, half, AscendC::ReduceOrder::ORDER_ONLY_VALUE>(lm_ubuf_tensor,
                                    ls_ubuf_tensor[offset],
                                    sub_m,                    // repeat
                                    1,                        // dstRepeatStride
                                    1,                        // srcBlockStride
                                    qk_round_n / BLOCK_SIZE_D   // srcRepeatStride
                                );
                                SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                            } else {
                                ub_to_ub<ArchType::ASCEND_V220, half>(
                                    ls32_ubuf_tensor.ReinterpretCast<half>(),
                                    ls_ubuf_tensor[offset],
                                    0,                                        // sid
                                    sub_m,                                    // nBurst
                                    VECTOR_SIZE_D / BLOCK_SIZE_D,                 // lenBurst
                                    (qk_round_n - VECTOR_SIZE_D) / BLOCK_SIZE_D,  // srcGap
                                    0                                         // dstGap
                                );
                                PIPE_BARRIER(V);
                                set_mask_d(qk_n - VECTOR_SIZE_D);
                                max_v<ArchType::ASCEND_V220, half>(ls32_ubuf_tensor.ReinterpretCast<half>(),
                                    ls32_ubuf_tensor.ReinterpretCast<half>(),
                                    ls_ubuf_tensor[offset + VECTOR_SIZE_D],
                                    sub_m,                   // repeat
                                    1,                       // dstBlockStride
                                    1,                       // src0BlockStride
                                    1,                       // src1BlockStride
                                    BLOCK_NUM_IN_REPEAT,                       // dstRepeatStride
                                    BLOCK_NUM_IN_REPEAT,                       // src0RepeatStride
                                    qk_round_n / BLOCK_SIZE_D  // src1RepeatStride
                                );
                                PIPE_BARRIER(V);
                                SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                                cmax_v<ArchType::ASCEND_V220, half, AscendC::ReduceOrder::ORDER_ONLY_VALUE>(lm_ubuf_tensor,
                                    ls32_ubuf_tensor.ReinterpretCast<half>(),
                                    sub_m,      // repeat
                                    1,          // dstRepeatStride
                                    1,          // srcBlockStride
                                    BLOCK_NUM_IN_REPEAT           // srcRepeatStride
                                );
                            }
                            PIPE_BARRIER(V);
                            if ((n_idx + split_idx) == 0) {
                                // *** hm = lm
                                ub_to_ub<ArchType::ASCEND_V220, half>(
                                    hm_ubuf_tensor,
                                    lm_ubuf_tensor,
                                    0,                         // sid
                                    1,                         // nBurst
                                    round_sub_m / BLOCK_SIZE_D,  // lenBurst
                                    0,                         // srcGap
                                    0                          // dstGap
                                );
                                PIPE_BARRIER(V);
                            } else {
                                // *** hm = MAX(lm, gm)
                                max_v<ArchType::ASCEND_V220, half>(hm_ubuf_tensor,
                                    lm_ubuf_tensor,
                                    gm_ubuf_tensor,
                                    sub_m_d128,  // repeat
                                    1,           // dstBlockStride
                                    1,           // src0BlockStride
                                    1,           // src1BlockStride
                                    BLOCK_NUM_IN_REPEAT,           // dstRepeatStride
                                    BLOCK_NUM_IN_REPEAT,           // src0RepeatStride
                                    BLOCK_NUM_IN_REPEAT            // src1RepeatStride
                                );
                                PIPE_BARRIER(V);
                                // *** dm = gm - hm
                                sub_v<ArchType::ASCEND_V220, half>(dm_ubuf_tensor[(n_idx + split_idx) % vect_mod * UB_HALF_LINE_SIZE_D],
                                    gm_ubuf_tensor,
                                    hm_ubuf_tensor,
                                    sub_m_d128,  // repeat
                                    1,           // dstBlockStride
                                    1,           // src0BlockStride
                                    1,           // src1BlockStride
                                    BLOCK_NUM_IN_REPEAT,           // dstRepeatStride
                                    BLOCK_NUM_IN_REPEAT,           // src0RepeatStride
                                    BLOCK_NUM_IN_REPEAT            // src1RepeatStride
                                );
                                PIPE_BARRIER(V);
                            }
                            // *** gm = hm
                            ub_to_ub<ArchType::ASCEND_V220, half>(
                                gm_ubuf_tensor,
                                hm_ubuf_tensor,
                                0,                         // sid
                                1,                         // nBurst
                                round_sub_m / BLOCK_SIZE_D,  // lenBurst
                                0,                         // srcGap
                                0                          // dstGap
                            );
                            PIPE_BARRIER(V);
                            // *** hm_block = expand_to_block(hm), 存放于 tv
                            brcb_v<ArchType::ASCEND_V220, uint16_t>(
                                tv_ubuf_tensor.ReinterpretCast<uint16_t>(),
                                hm_ubuf_tensor.template ReinterpretCast<uint16_t>(),
                                1,                              // dstBlockStride
                                BLOCK_NUM_IN_REPEAT,                              // dstRepeatStride
                                round_sub_m / FLOAT_BLOCK_SIZE_D  // repeat
                            );
                            PIPE_BARRIER(V);
                            // *** ls = ls - hm_block
                            for (uint32_t vsub_idx = 0; vsub_idx < qk_n / VECTOR_SIZE_D; ++vsub_idx) {
                                sub_v<ArchType::ASCEND_V220, half>(ls_ubuf_tensor[offset + vsub_idx * VECTOR_SIZE_D],
                                    ls_ubuf_tensor[offset + vsub_idx * VECTOR_SIZE_D],
                                    tv_ubuf_tensor.ReinterpretCast<half>(),
                                    sub_m,                    // repeat
                                    1,                        // dstBlockStride
                                    1,                        // src0BlockStride
                                    0,                        // src1BlockStride
                                    qk_round_n / BLOCK_SIZE_D,  // dstRepeatStride
                                    qk_round_n / BLOCK_SIZE_D,  // src0RepeatStride
                                    1                         // src1RepeatStride
                                );
                            }
                            if (qk_n % VECTOR_SIZE_D > 0) {
                                set_mask_d(qk_n % VECTOR_SIZE_D);
                                sub_v<ArchType::ASCEND_V220, half>(ls_ubuf_tensor[offset + qk_n / VECTOR_SIZE_D * VECTOR_SIZE_D],
                                    ls_ubuf_tensor[offset + qk_n / VECTOR_SIZE_D * VECTOR_SIZE_D],
                                    tv_ubuf_tensor.ReinterpretCast<half>(),
                                    sub_m,                    // repeat
                                    1,                        // dstBlockStride
                                    1,                        // src0BlockStride
                                    0,                        // src1BlockStride
                                    qk_round_n / BLOCK_SIZE_D,  // dstRepeatStride
                                    qk_round_n / BLOCK_SIZE_D,  // src0RepeatStride
                                    1                         // src1RepeatStride
                                );
                                SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                            }
                            PIPE_BARRIER(V);
                            // *** ls = castfp16to32(ls)
                            conv_v<ArchType::ASCEND_V220, half, float>(ls32_ubuf_tensor,
                                ls_ubuf_tensor[offset],
                                (sub_m * qk_round_n + FLOAT_VECTOR_SIZE_D - 1) / FLOAT_VECTOR_SIZE_D,  // repeat
                                1,                                                                 // dstBlockStride
                                1,                                                                 // srcBlockStride
                                BLOCK_NUM_IN_REPEAT,                                                                 // dstRepeatStride
                                BLOCK_NUM_IN_REPEAT_HALF                                                                  // srcRepeatStride
                            );
                            PIPE_BARRIER(V);
                            // *** ls = exp(ls)
                            exp_v<ArchType::ASCEND_V220, float>(ls32_ubuf_tensor,
                                ls32_ubuf_tensor,
                                (sub_m * qk_round_n + FLOAT_VECTOR_SIZE_D - 1) / FLOAT_VECTOR_SIZE_D,  // repeat
                                1,                                                                 // dstBlockStride
                                1,                                                                 // srcBlockStride
                                BLOCK_NUM_IN_REPEAT,                                                                 // dstRepeatStride
                                BLOCK_NUM_IN_REPEAT                                                                  // srcRepeatStride
                            );
                            PIPE_BARRIER(V);
                            // *** lp = castfp32to16(ls)
                            conv_v<ArchType::ASCEND_V220, float, P_DATA_TYPE>(lp_ubuf_tensor[offset],
                                ls32_ubuf_tensor,
                                (sub_m * qk_round_n + FLOAT_VECTOR_SIZE_D - 1) / FLOAT_VECTOR_SIZE_D,  // repeat
                                1,                                                                 // dstBlockStride
                                1,                                                                 // srcBlockStride
                                BLOCK_NUM_IN_REPEAT_HALF,                                                                 // dstRepeatStride
                                BLOCK_NUM_IN_REPEAT                                                                  // srcRepeatStride
                            );
                            PIPE_BARRIER(V);
                            SET_FLAG(V, MTE3, EVENT_ID0);
                            // *** ll = rowsum(ls32)
                            if (qk_n <= FLOAT_VECTOR_SIZE_D) {
                                set_mask_d(qk_n);
                                cadd_v<ArchType::ASCEND_V220, float>(ll_ubuf_tensor[(n_idx + split_idx) % vect_mod * UB_FLOAT_LINE_SIZE_D],
                                    ls32_ubuf_tensor,
                                    sub_m,                          // repeat
                                    1,                              // dstRepeatStride
                                    1,                              // srcBlockStride
                                    qk_round_n / FLOAT_BLOCK_SIZE_D   // srcRepeatStride
                                );
                                SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                            } else {
                                for (uint32_t rowsum_idx = 1; rowsum_idx < qk_n / FLOAT_VECTOR_SIZE_D; ++rowsum_idx) {
                                    add_v<ArchType::ASCEND_V220, float>(ls32_ubuf_tensor,
                                        ls32_ubuf_tensor,
                                        ls32_ubuf_tensor[rowsum_idx * FLOAT_VECTOR_SIZE_D],
                                        sub_m,                          // repeat
                                        1,                              // dstBlockStride
                                        1,                              // src0BlockStride
                                        1,                              // src1BlockStride
                                        qk_round_n / FLOAT_BLOCK_SIZE_D,  // dstRepeatStride
                                        qk_round_n / FLOAT_BLOCK_SIZE_D,  // src0RepeatStride
                                        qk_round_n / FLOAT_BLOCK_SIZE_D   // src1RepeatStride
                                    );
                                    PIPE_BARRIER(V);
                                }
                                if (qk_n % FLOAT_VECTOR_SIZE_D > 0) {
                                    set_mask_d(qk_n % FLOAT_VECTOR_SIZE_D);
                                    add_v<ArchType::ASCEND_V220, float>(ls32_ubuf_tensor,
                                        ls32_ubuf_tensor,
                                        ls32_ubuf_tensor[qk_n / FLOAT_VECTOR_SIZE_D * FLOAT_VECTOR_SIZE_D],
                                        sub_m,                          // repeat
                                        1,                              // dstBlockStride
                                        1,                              // src0BlockStride
                                        1,                              // src1BlockStride
                                        qk_round_n / FLOAT_BLOCK_SIZE_D,  // dstRepeatStride
                                        qk_round_n / FLOAT_BLOCK_SIZE_D,  // src0RepeatStride
                                        qk_round_n / FLOAT_BLOCK_SIZE_D   // src1RepeatStride
                                    );
                                    SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                                }
                                PIPE_BARRIER(V);
                                cadd_v<ArchType::ASCEND_V220, float>(ll_ubuf_tensor[(n_idx + split_idx) % vect_mod * UB_FLOAT_LINE_SIZE_D],
                                    ls32_ubuf_tensor,
                                    sub_m,                          // repeat
                                    1,                              // dstRepeatStride
                                    1,                              // srcBlockStride
                                    qk_round_n / FLOAT_BLOCK_SIZE_D   // srcRepeatStride
                                );
                            }
                            PIPE_BARRIER(V);
                            WAIT_FLAG(V, MTE3, EVENT_ID0);

                            ub_to_gm<ArchType::ASCEND_V220, P_DATA_TYPE>(
                                p_gm_tensor[(uint64_t)blockIdx_ * TMP_SIZE_64K + (n_idx + split_idx) % vect_mod * TMP_SIZE_64K / vect_mod +
                                    (uint64_t)subBlockIdx_ * qk_m / AIC_AIV_RATIO * qk_round_n],
                                lp_ubuf_tensor[offset],
                                0,                                // sid
                                1,                                // nBurst
                                sub_m * qk_round_n / BLOCK_SIZE_D,  // lenBurst
                                0,                                // srcGap
                                0                                 // dstGap
                            );
                            SET_FLAG(MTE3, MTE2, pingpong_flag);
                        }
                    }
                    if ASCEND_IS_AIV {
                        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_MTE3>(SOFTMAX_READY_D);
                    }
                }
                if (n_idx >= KV_INC) {
                    if ASCEND_IS_AIV {
                        AscendC::CrossCoreWaitFlag(UPDATE_READY_D);
                    }
                }
                for (uint32_t split_idx = 0; split_idx < KV_INC && n_idx + split_idx < n_end + KV_INC; split_idx++) {
                    if (n_idx + split_idx >= KV_INC && sub_m > 0) {
                        WAIT_FLAG(V, MTE2, EVENT_ID1);
                        pingpong_flag = (n_idx + split_idx + KV_INC) % DOUBLE_BUFFER_NUM;
                        gm_to_ub<ArchType::ASCEND_V220, float>(
                            lo_ubuf_tensor,
                            o_tmp_gm_tensor[(uint64_t)blockIdx_ * TMP_SIZE_64K + (n_idx + split_idx + KV_INC) % vect_mod * TMP_SIZE_64K / vect_mod +
                                (uint64_t)subBlockIdx_ * qk_m / AIC_AIV_RATIO * round_k],
                            0,                                   // sid
                            1,                                   // nBurst
                            sub_m * round_k / FLOAT_BLOCK_SIZE_D,  // lenBurst
                            0,                                   // srcGap
                            0                                    // dstGap
                        );
                        SET_FLAG(MTE2, V, EVENT_ID0);
                        WAIT_FLAG(MTE2, V, EVENT_ID0);
                        // *** 更新 L 和 O
                        if ((n_idx + split_idx) != KV_INC) {
                            // *** dm32 = castfp16to32(dm), 存放于 tv
                            conv_v<ArchType::ASCEND_V220, half, float>(tv_ubuf_tensor,
                                dm_ubuf_tensor[(n_idx + split_idx + KV_INC) % vect_mod * UB_HALF_LINE_SIZE_D],
                                sub_m_d64,  // repeat
                                1,          // dstBlockStride
                                1,          // srcBlockStride
                                BLOCK_NUM_IN_REPEAT,          // dstRepeatStride
                                BLOCK_NUM_IN_REPEAT_HALF           // srcRepeatStride
                            );
                            PIPE_BARRIER(V);
                            // *** dm_block = expand_to_block(dm), 存放于 tv
                            brcb_v<ArchType::ASCEND_V220, uint32_t>(tv_ubuf_tensor.ReinterpretCast<uint32_t>()[VECTOR_SIZE_D],
                                tv_ubuf_tensor.ReinterpretCast<uint32_t>(),
                                1,                              // dstBlockStride
                                BLOCK_NUM_IN_REPEAT,                              // dstRepeatStride
                                round_sub_m / FLOAT_BLOCK_SIZE_D  // repeat
                            );
                            PIPE_BARRIER(V);
                            // *** dm = exp(dm)
                            exp_v<ArchType::ASCEND_V220, float>(tv_ubuf_tensor,
                                tv_ubuf_tensor,
                                sub_m_d64,  // repeat
                                1,          // dstBlockStride
                                1,          // srcBlockStride
                                BLOCK_NUM_IN_REPEAT,          // dstRepeatStride
                                BLOCK_NUM_IN_REPEAT           // srcRepeatStride
                            );
                            PIPE_BARRIER(V);
                            // *** gl = dm * gl
                            mul_v<ArchType::ASCEND_V220, float>(gl_ubuf_tensor,
                                tv_ubuf_tensor,
                                gl_ubuf_tensor,
                                sub_m_d64,  // repeat
                                1,          // dstBlockStride
                                1,          // src0BlockStride
                                1,          // src1BlockStride
                                BLOCK_NUM_IN_REPEAT,          // dstRepeatStride
                                BLOCK_NUM_IN_REPEAT,          // src0RepeatStride
                                BLOCK_NUM_IN_REPEAT           // src1RepeatStride
                            );
                            PIPE_BARRIER(V);
                            // *** gl = ll + gl
                            add_v<ArchType::ASCEND_V220, float>(gl_ubuf_tensor,
                                gl_ubuf_tensor,
                                ll_ubuf_tensor[(n_idx + split_idx + KV_INC) % vect_mod * UB_FLOAT_LINE_SIZE_D],
                                sub_m_d64,  // repeat
                                1,          // dstBlockStride
                                1,          // src0BlockStride
                                1,          // src1BlockStride
                                BLOCK_NUM_IN_REPEAT,          // dstRepeatStride
                                BLOCK_NUM_IN_REPEAT,          // src0RepeatStride
                                BLOCK_NUM_IN_REPEAT           // src1RepeatStride
                            );
                            PIPE_BARRIER(V);
                            // *** dm_block = exp(dm_block)
                            exp_v<ArchType::ASCEND_V220, float>(tv_ubuf_tensor[VECTOR_SIZE_D],
                                tv_ubuf_tensor[VECTOR_SIZE_D],
                                (sub_m * FLOAT_BLOCK_SIZE_D + FLOAT_VECTOR_SIZE_D - 1) / FLOAT_VECTOR_SIZE_D,  // repeat
                                1,                                                                       // dstBlockStride
                                1,                                                                       // srcBlockStride
                                BLOCK_NUM_IN_REPEAT,                                                                       // dstRepeatStride
                                BLOCK_NUM_IN_REPEAT                                                                        // srcRepeatStride
                            );
                            PIPE_BARRIER(V);
                            if (go_flag_scalar == 1) {
                                WAIT_FLAG(MTE3, V, EVENT_ID0);
                                go_flag_scalar = 0;
                            }
                            // *** go = go * dm_block
                            for (uint32_t vmul_idx = 0; vmul_idx < headDim_ / FLOAT_VECTOR_SIZE_D; ++vmul_idx) {
                                mul_v<ArchType::ASCEND_V220, float>(go_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE_D],
                                    go_ubuf_tensor[vmul_idx * FLOAT_VECTOR_SIZE_D],
                                    tv_ubuf_tensor[VECTOR_SIZE_D],
                                    sub_m,                              // repeat
                                    BLK_STRIDE_ONE,                                  // dstBlockStride
                                    BLK_STRIDE_ONE,                                  // src0BlockStride
                                    0,                                  // src1BlockStride
                                    round_k / FLOAT_BLOCK_SIZE_D,       // dstRepeatStride
                                    round_k / FLOAT_BLOCK_SIZE_D,       // src0RepeatStride
                                    1                                   // src1RepeatStride
                                );
                            }
                            if (headDim_ % FLOAT_VECTOR_SIZE_D > 0) {
                                set_mask_d(headDim_ % FLOAT_VECTOR_SIZE_D);
                                mul_v<ArchType::ASCEND_V220, float>(go_ubuf_tensor[headDim_ / FLOAT_VECTOR_SIZE_D * FLOAT_VECTOR_SIZE_D],
                                    go_ubuf_tensor[headDim_ / FLOAT_VECTOR_SIZE_D * FLOAT_VECTOR_SIZE_D],
                                    tv_ubuf_tensor[VECTOR_SIZE_D],
                                    sub_m,                                  // repeat
                                    1,                                      // dstBlockStride
                                    1,                                      // src0BlockStride
                                    0,                                      // src1BlockStride
                                    round_k / FLOAT_BLOCK_SIZE_D,           // dstRepeatStride
                                    round_k / FLOAT_BLOCK_SIZE_D,           // src0RepeatStride
                                    1                                       // src1RepeatStride
                                );
                                SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                            }
                            PIPE_BARRIER(V);
                            // *** go = lo + go
                            add_v<ArchType::ASCEND_V220, float>(go_ubuf_tensor,
                                go_ubuf_tensor,
                                lo_ubuf_tensor,
                                (sub_m * round_k + FLOAT_VECTOR_SIZE_D - 1) / FLOAT_VECTOR_SIZE_D,  // repeat
                                1,                                                              // dstBlockStride
                                1,                                                              // src0BlockStride
                                1,                                                              // src1BlockStride
                                BLOCK_NUM_IN_REPEAT,                                                              // dstRepeatStride
                                BLOCK_NUM_IN_REPEAT,                                                              // src0RepeatStride
                                BLOCK_NUM_IN_REPEAT                                                               // src1RepeatStride
                            );
                            PIPE_BARRIER(V);
                        } else {
                            // *** gl = ll
                            ub_to_ub<ArchType::ASCEND_V220, float>(
                                gl_ubuf_tensor,
                                ll_ubuf_tensor[(n_idx + split_idx + KV_INC) % vect_mod * UB_FLOAT_LINE_SIZE_D],
                                0,                               // sid
                                1,                               // nBurst
                                round_sub_m / FLOAT_BLOCK_SIZE_D,  // lenBurst
                                0,                               // srcGap
                                0                                // dstGap
                            );
                            PIPE_BARRIER(V);
                            if (go_flag_scalar == 1) {
                                WAIT_FLAG(MTE3, V, EVENT_ID0);
                                go_flag_scalar = 0;
                            }
                            // *** go = lo
                            ub_to_ub<ArchType::ASCEND_V220, float>(
                                go_ubuf_tensor,
                                lo_ubuf_tensor,
                                0,                                   // sid
                                1,                                   // nBurst
                                sub_m * round_k / FLOAT_BLOCK_SIZE_D,  // lenBurst
                                0,                                   // srcGap
                                0                                    // dstGap
                            );
                            PIPE_BARRIER(V);
                        }
                        SET_FLAG(V, MTE2, EVENT_ID1);
                        if (n_idx + split_idx == (n_end + KV_INC - 1)) {
                            PIPE_BARRIER(V);
                            // *** gl_block = expand_to_block(gl), 存放于 tv
                            brcb_v<ArchType::ASCEND_V220, uint32_t>(tv_ubuf_tensor.ReinterpretCast<uint32_t>(),
                                gl_ubuf_tensor.ReinterpretCast<uint32_t>(),
                                1,                              // dstBlockStride
                                BLOCK_NUM_IN_REPEAT,                              // dstRepeatStride
                                round_sub_m / FLOAT_BLOCK_SIZE_D  // repeat
                            );
                            PIPE_BARRIER(V);
                            // *** go = go / gl_block
                            for (uint32_t vdiv_idx = 0; vdiv_idx < headDim_ / FLOAT_VECTOR_SIZE_D; ++vdiv_idx) {
                                div_v<ArchType::ASCEND_V220, float>(go_ubuf_tensor.ReinterpretCast<float>()[vdiv_idx * FLOAT_VECTOR_SIZE_D],
                                    go_ubuf_tensor.ReinterpretCast<float>()[vdiv_idx * FLOAT_VECTOR_SIZE_D],
                                    tv_ubuf_tensor.ReinterpretCast<float>(),
                                    sub_m,                 // repeat
                                    1,                     // dstBlockStride
                                    1,                     // src0BlockStride
                                    0,                     // src1BlockStride
                                    round_k / FLOAT_BLOCK_SIZE_D,  // dstRepeatStride
                                    round_k / FLOAT_BLOCK_SIZE_D,  // src0RepeatStride
                                    1                      // src1RepeatStride
                                );
                            }
                            if (headDim_ % FLOAT_VECTOR_SIZE_D > 0) {
                                set_mask_d(headDim_ % FLOAT_VECTOR_SIZE_D);
                                div_v<ArchType::ASCEND_V220, float>(go_ubuf_tensor.ReinterpretCast<float>()[headDim_ / FLOAT_VECTOR_SIZE_D * FLOAT_VECTOR_SIZE_D],
                                    go_ubuf_tensor.ReinterpretCast<float>()[headDim_ / FLOAT_VECTOR_SIZE_D * FLOAT_VECTOR_SIZE_D],
                                    tv_ubuf_tensor.ReinterpretCast<float>(),
                                    sub_m,                 // repeat
                                    BLK_STRIDE_ONE,                     // dstBlockStride
                                    BLK_STRIDE_ONE,                     // src0BlockStride
                                    0,                     // src1BlockStride
                                    round_k / FLOAT_BLOCK_SIZE_D,  // dstRepeatStride
                                    round_k / FLOAT_BLOCK_SIZE_D,  // src0RepeatStride
                                    1                      // src1RepeatStride
                                );
                                SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
                            }

                            PIPE_BARRIER(V);

                            conv_v<ArchType::ASCEND_V220, float, P_DATA_TYPE>(go_ubuf_tensor.ReinterpretCast<P_DATA_TYPE>(),
                                go_ubuf_tensor.ReinterpretCast<float>(),
                                (sub_m * round_k + FLOAT_VECTOR_SIZE_D - 1) / FLOAT_VECTOR_SIZE_D,  // repeat
                                1,          // dstBlockStride
                                1,          // srcBlockStride
                                BLOCK_NUM_IN_REPEAT_HALF,          // dstRepeatStride
                                BLOCK_NUM_IN_REPEAT           // srcRepeatStride
                            );
                            // ********************* move O to GM ************************
                            SET_FLAG(V, MTE3, EVENT_ID0);
                            WAIT_FLAG(V, MTE3, EVENT_ID0);
                            ub_to_gm_align<ArchType::ASCEND_V220, P_DATA_TYPE>(
                                o_gm_tensor[o_offset + (uint64_t)subBlockIdx_ * qk_m / AIC_AIV_RATIO * stride_qo],
                                go_ubuf_tensor.ReinterpretCast<P_DATA_TYPE>(),
                                0,                     // sid
                                sub_m,                 // nBurst
                                headDim_ * CONST_TWO,               // lenBurst
                                0,                     // leftPaddingNum
                                0,                     // rightPaddingNum
                                0,                     // srcGap
                                (stride_qo - headDim_) * CONST_TWO  // dstGap
                            );
                            if (go_flag_scalar == 0) {
                                SET_FLAG(MTE3, V, EVENT_ID0);
                                go_flag_scalar = 1;
                            }
                        }
                    }
                }
            }
        }
        WAIT_FLAG(MTE3, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE3, MTE2, EVENT_ID1);
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID1);
        WAIT_FLAG(V, MTE2, EVENT_ID2);
        WAIT_FLAG(MTE3, V, EVENT_ID0);
    }

    __aicore__ inline void set_mask_d(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = len % FLOAT_VECTOR_SIZE_D;
        for (int64_t i = 0; i < temp; i++) {
            mask |= one << i;
        }

        if (len == VECTOR_SIZE_D) {
            SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else if (len >= FLOAT_VECTOR_SIZE_D) {
            SetVectorMask<int8_t>(mask, (uint64_t)-1);
        } else {
            SetVectorMask<int8_t>(0x0, mask);
        }
    }

private:

    __gm__ uint8_t *__restrict__ tiling_para_gm;
    __gm__ uint8_t *__restrict__ mask_gm;
    __gm__ uint8_t *key_ = nullptr;
    __gm__ uint8_t *value_ = nullptr;

    AscendC::GlobalTensor<IN_KV_TYPE> k_gm_tensor;
    AscendC::GlobalTensor<IN_KV_TYPE> v_gm_tensor;
    AscendC::GlobalTensor<int64_t> actualSeqLenGmTensorQ;
    AscendC::GlobalTensor<int64_t> actualSeqLenGmTensorKv;
    AscendC::GlobalTensor<int32_t> blockTableGmTensor;

    AscendC::GlobalTensor<MASK_DATA_TYPE> mask_gm_tensor;
    AscendC::GlobalTensor<P_DATA_TYPE> o_gm_tensor;
    AscendC::GlobalTensor<S_DATA_TYPE> s_gm_tensor;
    AscendC::GlobalTensor<P_DATA_TYPE> p_gm_tensor;
    AscendC::GlobalTensor<float> o_tmp_gm_tensor;
    AscendC::GlobalTensor<DEQUANT_KV_TYPE> dequantKeyWsPing_;
    AscendC::GlobalTensor<DEQUANT_KV_TYPE> dequantKeyWsPong_;
    AscendC::GlobalTensor<DEQUANT_KV_TYPE> dequantValueWsPing_;
    AscendC::GlobalTensor<DEQUANT_KV_TYPE> dequantValueWsPong_;

    AscendC::GlobalTensor<float> antiquantScaleGmTensor_;
    AscendC::GlobalTensor<float> antiquantOffsetGmTensor_;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<int8_t> kvUbTensorInt8_;
    AscendC::LocalTensor<half> kvUbTensorHalf_;
    AscendC::LocalTensor<DEQUANT_KV_TYPE> kvUbTensorDequant_;
    AscendC::LocalTensor<float> kvUbTensorFp32_;
    AscendC::LocalTensor<int32_t> dequantOffsetUbTensor_;
    AscendC::LocalTensor<float> dequantScaleUbTensor_;
    AscendC::LocalTensor<float> dequantOffsetUbTensorFp32_;

    AscendC::LocalTensor<S_DATA_TYPE> ls_ubuf_tensor;
    AscendC::LocalTensor<P_DATA_TYPE> lp_ubuf_tensor;
    AscendC::LocalTensor<float> ls32_ubuf_tensor;
    AscendC::LocalTensor<S_DATA_TYPE> mask_ubuf_tensor;
    AscendC::LocalTensor<MASK_DATA_TYPE> mask16_ubuf_tensor;
    AscendC::LocalTensor<float> lo_ubuf_tensor;
    AscendC::LocalTensor<S_DATA_TYPE> lm_ubuf_tensor;
    AscendC::LocalTensor<S_DATA_TYPE> hm_ubuf_tensor;
    AscendC::LocalTensor<S_DATA_TYPE> gm_ubuf_tensor;
    AscendC::LocalTensor<S_DATA_TYPE> dm_ubuf_tensor;
    AscendC::LocalTensor<float> ll_ubuf_tensor;
    AscendC::LocalTensor<float> gl_ubuf_tensor;
    AscendC::LocalTensor<float> tv_ubuf_tensor;
    AscendC::LocalTensor<float> go_ubuf_tensor;

    uint32_t batchSize_;
    uint32_t qHeadNum_;
    uint32_t embedding_size;
    uint32_t kvHeadNum_;
    uint32_t max_context_len;
    S_DATA_TYPE tor;
    uint32_t kvBlockSize_;
    uint32_t totalBlockNumQ_;
    uint32_t is_triu_mask = 0;
    uint32_t tiling_head_size;
    uint32_t tiling_para_size;
    uint32_t batch_stride;
    uint32_t head_stride;
    uint32_t stride_qo;
    uint32_t headDim_;
    uint32_t round_k;
    uint32_t prefill_batch_size_;
    uint32_t decoder_batch_size_;
    uint32_t mask_type;
    uint32_t max_kv;
    uint32_t maxNumBlocksPerQuery;

    uint32_t numBlocks_;
    uint32_t blockIdx_;
    uint32_t seqStepQ_;
    uint32_t seqStepKv_;
    uint32_t kvAntiquantBiasFlag_ = 1;
    uint32_t groupNum = 0;
    uint32_t subBlockIdx_ = 0;
};

template <typename IFAT> class PagedAttentionAntiquant {
    public:

    __aicore__ inline PagedAttentionAntiquant(){}
    __aicore__ inline void InitQuant(__gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2,
                                     __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2,
                                     __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                                     __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset,
                                     __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *valueAntiquantOffset,
                                     __gm__ uint8_t *workspace)
    {
        this->antiquant_scale = antiquantScale;
        this->antiquant_offset = antiquantOffset;
    }
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ,
                                __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                const typename IFAT::TilingType *__restrict tiling)
    {
        this->q_gm = query;
        this->k_gm = key;
        this->v_gm = value;
        this->actual_seq_len_q_gm = actualSeqLengthsQ;
        this->actual_seq_len_gm = actualSeqLengths;
        this->mask_gm = attenMask;
        this->block_tables_gm = blockTable;
        this->o_gm = attentionOut;
        this->workspace_gm = workspace;
        this->tilingData = tiling;
    }
    __aicore__ inline void Process() {
        uint32_t decoder_batch_size = 0;
        uint32_t prefill_batch_size = 1;
        if ASCEND_IS_AIC {
            PagedAttentionParallelAic<IFAT, TilingKeyType::TILING_INT8_VEC_QUANT> prefill_pa_aic(prefill_batch_size, decoder_batch_size);
            prefill_pa_aic.SetArgs(q_gm, k_gm, v_gm, actual_seq_len_q_gm, actual_seq_len_gm, block_tables_gm, workspace_gm, tilingData);
            prefill_pa_aic.Run();
        }
            
        if ASCEND_IS_AIV {
            PagedAttentionParallelAiv<IFAT, TilingKeyType::TILING_INT8_VEC_QUANT> prefill_pa_aiv(prefill_batch_size, decoder_batch_size);
            prefill_pa_aiv.SetArgs(k_gm, v_gm, actual_seq_len_q_gm, actual_seq_len_gm, block_tables_gm, mask_gm, o_gm, workspace_gm, tilingData);
            prefill_pa_aiv.InitQuant(antiquant_scale, antiquant_offset);
            prefill_pa_aiv.Run();
        }      
    }
private:
    __gm__ uint8_t *__restrict__ q_gm;
    __gm__ uint8_t *__restrict__ k_gm;
    __gm__ uint8_t *__restrict__ v_gm;
    __gm__ uint8_t *__restrict__ actual_seq_len_gm;
    __gm__ uint8_t *__restrict__ actual_seq_len_q_gm;
    __gm__ uint8_t *__restrict__ block_tables_gm;
    __gm__ uint8_t *__restrict__ mask_gm;
    __gm__ uint8_t *__restrict__ antiquant_scale;
    __gm__ uint8_t *__restrict__ antiquant_offset;
    __gm__ uint8_t *__restrict__ scale_gm;
    __gm__ uint8_t *__restrict__ o_gm;
    __gm__ uint8_t *__restrict__ workspace_gm;
    const typename IFAT::TilingType *__restrict tilingData = nullptr;
};

#endif
