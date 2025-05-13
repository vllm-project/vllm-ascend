/*
 * Copyright (c) China Merchants Bank Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 1;
class KernelAdvanceStep{
public:
    __aicore__ inline KernelAdvanceStep() {}
    __aicore__ inline void Init(int32_t tasks_per_core,
                                int32_t num_queries,
                                __gm__ int64_t* input_tokens_ptr,
                                __gm__ int64_t* sampled_token_ids_ptr,
                                __gm__ int64_t* input_positions_ptr,
                                __gm__ int32_t* seq_lens_ptr,
                                __gm__ int32_t* slot_mapping_ptr)
    {
        this->tasks_per_core = tasks_per_core;

        this->start_id = this->tasks_per_core * AscendC::GetBlockIdx();
        this->end_id = this->tasks_per_core * (AscendC::GetBlockIdx() + 1) - 1;

        // actual task nums of each core
        this->actual_task_per_core = tasks_per_core;
        if(this->end_id >= num_queries) {
            this->actual_task_per_core = num_queries - this->start_id;
            this->end_id = num_queries - 1;
        }

        int32_t offset_this_core = this->tasks_per_core * AscendC::GetBlockIdx();

        // init outQues
        pipe.InitBuffer(outQueInputTokens, BUFFER_NUM, this->actual_task_per_core * sizeof(int64_t));
        pipe.InitBuffer(outQueInputPos, BUFFER_NUM, this->actual_task_per_core * sizeof(int64_t));
        pipe.InitBuffer(outQueSeqLen, BUFFER_NUM, this->actual_task_per_core * sizeof(int32_t));
        pipe.InitBuffer(outQueSlotMapping, BUFFER_NUM, this->actual_task_per_core * sizeof(int32_t));

        // init inQues
        pipe.InitBuffer(inQueSeqLen,BUFFER_NUM, this->actual_task_per_core * sizeof(int32_t));
        pipe.InitBuffer(inQueSampledTokenIds,BUFFER_NUM, this->actual_task_per_core * sizeof(int64_t));

        // init GlobalMemory
        inputTokensGm.SetGlobalBuffer((__gm__ int64_t *)input_tokens_ptr + offset_this_core, this->actual_task_per_core);
        sampledTokenIdsGm.SetGlobalBuffer((__gm__ int64_t *)sampled_token_ids_ptr + offset_this_core, this->actual_task_per_core);
        inputPositionsGm.SetGlobalBuffer((__gm__ int64_t *)input_positions_ptr + offset_this_core, this->actual_task_per_core);
        seqLensGm.SetGlobalBuffer((__gm__ int32_t *)seq_lens_ptr + offset_this_core, this->actual_task_per_core);
        slotMappingGm.SetGlobalBuffer((__gm__ int32_t *)slot_mapping_ptr + offset_this_core, this->actual_task_per_core);
    }
    __aicore__ inline void Process(int64_t block_size, __gm__ int32_t* block_tables_ptr,  int64_t block_tables_stride)
    {
        // no need for tilling or pipeline parallel within each core, as the amount of data processed is very small
        CopyIn();
        Update(block_size, block_tables_ptr, block_tables_stride);
        CopyOut();
    }

private:
     __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<int32_t> seqLenLocalIn = inQueSeqLen.AllocTensor<int32_t>();
        AscendC::LocalTensor<int64_t> sampledTokenIdsLocal = inQueSampledTokenIds.AllocTensor<int64_t>();

        AscendC::DataCopyExtParams copyParams32{1, static_cast<uint32_t>(this->actual_task_per_core * sizeof(int32_t)), 0, 0, 0}; // blockLen = tasks_per_core * 32 / 8 个字节（int32为4字节)
        AscendC::DataCopyExtParams copyParams64{1, static_cast<uint32_t>(this->actual_task_per_core * sizeof(int64_t)), 0, 0, 0}; // blockLen = tasks_per_core * 64 / 8 个字节（int64为8字节）

        // calculate the nums that need padded
        // so that the total length becomes a multiple of 32 bytes which is a requirement of DataCopy Function.
        uint8_t remainNum32 =this->actual_task_per_core * sizeof(int32_t) % 32;
        uint8_t needPadElements32 = remainNum32 == 0 ? remainNum32 : (32 - remainNum32) / sizeof(int32_t);

        AscendC::DataCopyPadExtParams<int32_t> padParams32{true, 0, needPadElements32, 0};

        // calculate the nums that need padded
        // so that the total length becomes a multiple of 32 bytes which is a requirement of DataCopy Function.
        uint8_t remainNum64 =this->actual_task_per_core * sizeof(int64_t) % 32;
        uint8_t needPadElements64 = remainNum64 == 0 ? remainNum64 : (32 - remainNum64) / sizeof(int64_t);
        AscendC::DataCopyPadExtParams<int64_t> padParams64{true, 0, needPadElements64, 0};

        AscendC::DataCopyPad(seqLenLocalIn, seqLensGm, copyParams32, padParams32);
        AscendC::DataCopyPad(sampledTokenIdsLocal, sampledTokenIdsGm, copyParams64, padParams64);

        inQueSeqLen.EnQue(seqLenLocalIn);
        inQueSampledTokenIds.EnQue(sampledTokenIdsLocal);
    }
    __aicore__ inline void Update(int64_t block_size, __gm__ int32_t* block_tables_ptr, int64_t block_tables_stride)
    {
        // input
        AscendC::LocalTensor<int32_t> seqLenLocalIn = inQueSeqLen.DeQue<int32_t>();
        AscendC::LocalTensor<int64_t> sampledTokenIdsLocal = inQueSampledTokenIds.DeQue<int64_t>();

        // output
        AscendC::LocalTensor<int64_t> inputTokensLocal = outQueInputTokens.AllocTensor<int64_t>();
        AscendC::LocalTensor<int64_t> inputPosLocal = outQueInputPos.AllocTensor<int64_t>();
        AscendC::LocalTensor<int32_t> seqLenLocalOut = outQueSeqLen.AllocTensor<int32_t>();
        AscendC::LocalTensor<int32_t> slotMappingLocal = outQueSlotMapping.AllocTensor<int32_t>();

        auto unary_params = AscendC::UnaryRepeatParams(1, 1, 8, 8);

        //Use "for" instead of AscendC::Adds function because AscendC::Adds does not work
        //when srcLocalMemory has different datatype from dstLocalMemory
        for(int i=0; i < this->actual_task_per_core; i++) {
            inputTokensLocal.SetValue(i, sampledTokenIdsLocal.GetValue(i));
            inputPosLocal.SetValue(i, seqLenLocalIn.GetValue(i));
        }

        AscendC::Adds<int32_t, false>(seqLenLocalOut, seqLenLocalIn, 1, (uint64_t)0, 1, unary_params);

        // Gather blockTables with dim=1, block_index. No Ascend Function available, use "for" instead.
        for(int cur_query_id = this->start_id, i = 0; i < this->actual_task_per_core; cur_query_id++, i++) {
            __gm__ int32_t const* seq_block_tables_ptr = block_tables_ptr + block_tables_stride * cur_query_id;

            int block_index = inputPosLocal.GetValue(i) / block_size;
            int block_offset = inputPosLocal.GetValue(i) % block_size;

            int slot_num = seq_block_tables_ptr[block_index] * block_size + block_offset;
            // Update slot_mapping
            slotMappingLocal.SetValue(i,slot_num);
        }

        outQueInputTokens.EnQue(inputTokensLocal);
        outQueInputPos.EnQue(inputPosLocal);
        outQueSeqLen.EnQue(seqLenLocalOut);
        outQueSlotMapping.EnQue(slotMappingLocal);

        inQueSampledTokenIds.FreeTensor(sampledTokenIdsLocal);
        inQueSeqLen.FreeTensor(seqLenLocalIn);

    }
    __aicore__ inline void CopyOut()
    {
        AscendC::DataCopyExtParams copyParams32{1, static_cast<uint32_t>(this->actual_task_per_core * sizeof(int32_t)),0,0,0};
        AscendC::DataCopyExtParams copyParams64{1, static_cast<uint32_t>(this->actual_task_per_core * sizeof(int64_t)),0,0,0};

        AscendC::LocalTensor<int64_t> inputTokensLocal = outQueInputTokens.DeQue<int64_t>();
        AscendC::DataCopyPad(inputTokensGm, inputTokensLocal, copyParams64);
        outQueInputTokens.FreeTensor(inputTokensLocal);

        AscendC::LocalTensor<int64_t> inputPosLocal = outQueInputPos.DeQue<int64_t>();
        AscendC::DataCopyPad(inputPositionsGm, inputPosLocal, copyParams64);
        outQueInputPos.FreeTensor(inputPosLocal);

        AscendC::LocalTensor<int32_t> seqLenLocalOut = outQueSeqLen.DeQue<int32_t>();
        AscendC::DataCopyPad(seqLensGm, seqLenLocalOut, copyParams32);
        outQueSeqLen.FreeTensor(seqLenLocalOut);

        AscendC::LocalTensor<int32_t> slotMappingLocal = outQueSlotMapping.DeQue<int32_t>();
        AscendC::DataCopyPad(slotMappingGm, slotMappingLocal, copyParams32);
        outQueSlotMapping.FreeTensor(slotMappingLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueInputTokens, outQueInputPos,
                                                            outQueSeqLen, outQueSlotMapping;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueSeqLen,
                                                           inQueSampledTokenIds,
                                                           inQueBlockTables;

    AscendC::GlobalTensor<int64_t> inputTokensGm, sampledTokenIdsGm, inputPositionsGm ;

    AscendC::GlobalTensor<int32_t> seqLensGm, slotMappingGm, blockTablesGm;

    int32_t tasks_per_core, start_id, end_id, actual_task_per_core;
};

extern "C" __global__ __aicore__ void AdvanceStepFlashAttnKernel(
    int64_t num_seqs,
    int64_t num_queries,
    int64_t block_size,
    __gm__ int64_t* input_tokens_ptr,
    __gm__ int64_t* sampled_token_ids_ptr,
    __gm__ int64_t* input_positions_ptr,
    __gm__ int32_t* seq_lens_ptr,
    __gm__ int32_t* slot_mapping_ptr,
    __gm__ int32_t* block_tables_ptr,
    int64_t block_tables_stride,
    int32_t tasks_per_core
)
{
    int start_id = tasks_per_core * AscendC::GetBlockIdx();
    // no task for this core.
    if(start_id >= num_queries) {
        return;
    }
    KernelAdvanceStep advanceStep;
    advanceStep.Init(tasks_per_core, num_queries, input_tokens_ptr, sampled_token_ids_ptr, input_positions_ptr, seq_lens_ptr, slot_mapping_ptr);
    advanceStep.Process(block_size,block_tables_ptr,block_tables_stride);
}

namespace vllm_ascend
{

extern void launch_advance_step_flashattn(
    void* stream,
    int64_t num_seqs,
    int64_t num_queries,
    int64_t block_size,
    int64_t* input_tokens_ptr,
    int64_t* sampled_token_ids_ptr,
    int64_t* input_positions_ptr,
    int32_t* seq_lens_ptr,
    int32_t* slot_mapping_ptr,
    int32_t* block_tables_ptr,
    int64_t block_tables_stride)
{
    int32_t num_cores = 20;

    if(num_cores > num_queries) {
        num_cores = num_queries;
    }

    // task num processed of each core
    int32_t tasks_per_core = (num_queries + num_cores - 1) / num_cores;

    AdvanceStepFlashAttnKernel<<<num_cores, nullptr, stream>>>(
        num_seqs,
        num_queries,
        block_size,
        input_tokens_ptr,
        sampled_token_ids_ptr,
        input_positions_ptr,
        seq_lens_ptr,
        slot_mapping_ptr,
        block_tables_ptr,
        block_tables_stride,
        tasks_per_core);
}

}
