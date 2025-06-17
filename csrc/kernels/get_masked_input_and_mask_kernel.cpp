/* 
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#include "kernel_operator.h"
#include "kernel_tensor_impl.h"
#include "kernel_type.h"
#include "types.h"
#include "utils.h"
using vllm_ascend::AccType;

template<typename scalar_t>
class GetMaskedInputAndMask {
public:
    __aicore__ inline GetMaskedInputAndMask() {}
    
    __aicore__ inline ~GetMaskedInputAndMask() {
        pipe.Reset();
    }

    
    __aicore__ inline void Init(
        __gm__ scalar_t* input,
        __gm__ scalar_t* masked_input, 
        __gm__ bool* mask_out,
        const int64_t org_vocab_start_index,
        const int64_t org_vocab_end_index,
        const int64_t num_org_vocab_padding,
        const int64_t added_vocab_start_index,
        const int64_t added_vocab_end_index,
        const int64_t size)
    {
        // Initialize basic parameters
        input_ = input;
        masked_input_ = masked_input;
        mask_out_ = mask_out;
        org_vocab_start_index_ = org_vocab_start_index;
        org_vocab_end_index_ = org_vocab_end_index;
        size_ = ((size + 31) / 32) * 32;
        added_offset_ = added_vocab_start_index - 
            (org_vocab_end_index - org_vocab_start_index) - 
            num_org_vocab_padding;
        added_vocab_start_index_ = added_vocab_start_index;
        added_vocab_end_index_ = added_vocab_end_index;

        // Initialize global tensors
        inputGlobal.SetGlobalBuffer(input);
        maskedOutputGlobal.SetGlobalBuffer(masked_input); 
        maskOutGlobal.SetGlobalBuffer(mask_out);

        // Initialize queues
        pipe.InitBuffer(inQueue, 1, size_ * sizeof(scalar_t));
        pipe.InitBuffer(outQueue, 1, size_ * sizeof(scalar_t));
        pipe.InitBuffer(maskQueue, 1, size_ * sizeof(bool));
        
        // Initialize calculation buffers
        pipe.InitBuffer(calc_buf_1, size_ * sizeof(float));
        pipe.InitBuffer(calc_buf_2, size_ * sizeof(float));
        
        // Initialize result queues
        pipe.InitBuffer(result_ge_que, BUFFER_NUM, size_ * sizeof(float));
        pipe.InitBuffer(result_le_que, BUFFER_NUM, size_ * sizeof(float));
        pipe.InitBuffer(result_org_mask_que, BUFFER_NUM, size_ * sizeof(float));
        pipe.InitBuffer(result_add_mask_que, BUFFER_NUM, size_ * sizeof(float));

        // Initialize temporary buffers
        pipe.InitBuffer(start_buf, size_ * sizeof(float));
        pipe.InitBuffer(end_buf, size_ * sizeof(float));
        pipe.InitBuffer(inputFloat_buf, size_ * sizeof(float));
        pipe.InitBuffer(validOffset_buf, size_ * sizeof(float));
        pipe.InitBuffer(vocabMask_buf_, size_ * sizeof(int8_t));
        pipe.InitBuffer(ones_buf_, size_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<scalar_t> inputLocal = inQueue.AllocTensor<scalar_t>();
        AscendC::DataCopy(inputLocal, inputGlobal, size_);
        inQueue.EnQue(inputLocal);
    }

    __aicore__ inline void CompareWithValue(
        AscendC::LocalTensor<int8_t>& result,
        const AscendC::LocalTensor<float>& input,
        const AscendC::LocalTensor<float>& compare_value,
        bool is_greater_equal) {

        AscendC::LocalTensor<float> compute_buf = calc_buf_1.Get<float>();
        if (is_greater_equal) {
            AscendC::Max(compute_buf, input, compare_value, size_);  
            AscendC::Sub(compute_buf, compare_value, compute_buf, size_);  
        } else {
            AscendC::Max(compute_buf, input, compare_value, size_); 
            AscendC::Sub(compute_buf, compute_buf, compare_value, size_); 
        }

        AscendC::Abs(compute_buf, compute_buf, size_);
        AscendC::Mins(compute_buf, compute_buf, MIN_ACCURACY_FP32, size_);
        AscendC::Muls(compute_buf, compute_buf, MAX_MUL_1_FP32, size_);
        AscendC::Muls(compute_buf, compute_buf, MAX_MUL_1_FP32, size_);
        AscendC::Muls(compute_buf, compute_buf, MAX_MUL_2_FP32, size_);
        AscendC::Adds(compute_buf, compute_buf, NEGATIVE_ONE_FP32, size_);
        AscendC::Abs(compute_buf, compute_buf, size_);

        AscendC::LocalTensor<half> compute_buf_fp16 = calc_buf_2.Get<half>();
        AscendC::Cast(compute_buf_fp16, compute_buf, AscendC::RoundMode::CAST_NONE, size_);
        AscendC::Cast(result, compute_buf_fp16, AscendC::RoundMode::CAST_NONE, size_);
    }

    __aicore__ inline void ComputeRangeMask(
        AscendC::LocalTensor<int8_t>& range_mask,
        const AscendC::LocalTensor<float>& input,
        const float start_value, 
        const float end_value) {
        
        // Use already initialized buffers
        AscendC::LocalTensor<float> start_value_tensor = start_buf.Get<float>();
        AscendC::LocalTensor<float> end_value_tensor = end_buf.Get<float>();

        AscendC::Duplicate(start_value_tensor, start_value, size_);
        AscendC::Duplicate(end_value_tensor, end_value, size_);
        
        AscendC::LocalTensor<int8_t> ge_result = result_ge_que.AllocTensor<int8_t>();
        AscendC::LocalTensor<int8_t> lt_result = result_le_que.AllocTensor<int8_t>();

        CompareWithValue(ge_result, start_value_tensor, input, true);
        CompareWithValue(lt_result, input, end_value_tensor, false);
        
        AscendC::And(range_mask, ge_result, lt_result, size_);
    }

    __aicore__ inline void Compute() {
        AscendC::LocalTensor<scalar_t> inputLocal = inQueue.DeQue<scalar_t>();
        AscendC::LocalTensor<scalar_t> maskedLocal = outQueue.AllocTensor<scalar_t>();
        AscendC::LocalTensor<int8_t> maskLocal = maskQueue.AllocTensor<int8_t>();

        AscendC::LocalTensor<float> inputFloat = inputFloat_buf.Get<float>();
        AscendC::Cast(inputFloat, inputLocal, AscendC::RoundMode::CAST_NONE, size_);

        // Calculate mask for org_vocab range
        // org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
        AscendC::LocalTensor<int8_t> orgVocabMask = result_org_mask_que.AllocTensor<int8_t>();
        ComputeRangeMask(orgVocabMask, 
                        inputFloat,
                        static_cast<float>(org_vocab_start_index_),
                        static_cast<float>(org_vocab_end_index_));

        // Calculate mask for added_vocab range
        // added_vocab_mask = (input_ >= added_vocab_start_index) & (input_ < added_vocab_end_index)
        AscendC::LocalTensor<int8_t> addedVocabMask = result_add_mask_que.AllocTensor<int8_t>();
        ComputeRangeMask(addedVocabMask,
                        inputFloat,
                        static_cast<float>(added_vocab_start_index_),
                        static_cast<float>(added_vocab_end_index_));

        // Calculate validOffset
        // valid_offset = (org_vocab_start_index * org_vocab_mask) + (added_offset * added_vocab_mask)
        AscendC::LocalTensor<float> validOffset = validOffset_buf.Get<float>();
        AscendC::LocalTensor<float> constOrgStartIndex = start_buf.Get<float>();
        
        AscendC::Duplicate(constOrgStartIndex, float(org_vocab_start_index_), size_);
        
        AscendC::LocalTensor<half> orgVocabMask_fp16;
        AscendC::LocalTensor<float> orgVocabMask_fp32;
        AscendC::Cast(orgVocabMask_fp16, orgVocabMask, AscendC::RoundMode::CAST_NONE, size_);
        AscendC::Cast(orgVocabMask_fp32, orgVocabMask_fp16, AscendC::RoundMode::CAST_NONE, size_);

        AscendC::Mul(validOffset, 
            constOrgStartIndex,
            orgVocabMask_fp32,
            size_);

        AscendC::LocalTensor<float> addedOffset;
        AscendC::LocalTensor<float> addedOffsetTensor = end_buf.Get<float>();
        AscendC::Duplicate(addedOffsetTensor, float(added_offset_), size_);

        AscendC::LocalTensor<half> addedVocabMask_fp16;
        AscendC::LocalTensor<float> addedVocabMask_fp32;
        AscendC::Cast(addedVocabMask_fp16, addedVocabMask, AscendC::RoundMode::CAST_NONE, size_);
        AscendC::Cast(addedVocabMask_fp32, addedVocabMask_fp16, AscendC::RoundMode::CAST_NONE, size_);

        AscendC::Mul(addedOffset, 
            addedOffsetTensor,
            addedVocabMask_fp32,
            size_);
            
        AscendC::Add(validOffset, validOffset, addedOffset, size_);

        // vocab_mask = org_vocab_mask | added_vocab_mask
        AscendC::LocalTensor<int8_t> vocabMask = vocabMask_buf_.Get<int8_t>();

        AscendC::Or(vocabMask,
                    orgVocabMask,
                    addedVocabMask,
                    size_);
                    
        AscendC::Sub(inputFloat, inputFloat, validOffset, size_);

        // input_ = vocab_mask * (input_ - valid_offset)
        AscendC::LocalTensor<half> vocabMask_fp16;
        AscendC::LocalTensor<float> vocabMask_fp32;
        AscendC::Cast(vocabMask_fp16, vocabMask, AscendC::RoundMode::CAST_NONE, size_);
        AscendC::Cast(vocabMask_fp32, vocabMask_fp16, AscendC::RoundMode::CAST_NONE, size_);
        
        AscendC::LocalTensor<float> inputFloat_fp32;
        AscendC::Mul(inputFloat, inputFloat, vocabMask_fp32, size_);

        AscendC::Cast(maskedLocal, inputFloat, AscendC::RoundMode::CAST_CEIL, size_);  
        outQueue.EnQue(maskedLocal);

        // ~vocab_mask
        AscendC::LocalTensor<float> ones_tensor = ones_buf_.Get<float>();
        AscendC::Duplicate(ones_tensor, (float)1, size_);
        AscendC::LocalTensor<float> maskLocal_fp32;

        AscendC::Sub(maskLocal_fp32, 
            ones_tensor,
            vocabMask_fp32,
            size_);

        AscendC::LocalTensor<half> maskLocal_fp16;
        AscendC::Cast(maskLocal_fp16, maskLocal_fp32, AscendC::RoundMode::CAST_NONE, size_);
        AscendC::Cast(maskLocal, maskLocal_fp16, AscendC::RoundMode::CAST_NONE, size_);
        maskQueue.EnQue(maskLocal);
        inQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<scalar_t> maskedLocal = outQueue.DeQue<scalar_t>();
        AscendC::LocalTensor<bool> maskLocal = maskQueue.DeQue<bool>();
        
        AscendC::DataCopy(maskedOutputGlobal, maskedLocal, size_);
        AscendC::DataCopy(maskOutGlobal, maskLocal, size_);
        
        outQueue.FreeTensor(maskedLocal);
        maskQueue.FreeTensor(maskLocal);
    }

private:
    static constexpr int32_t BUFFER_NUM = 2;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue, maskQueue;
    AscendC::GlobalTensor<scalar_t> inputGlobal, maskedOutputGlobal;
    AscendC::GlobalTensor<bool> maskOutGlobal;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> result_ge_que;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> result_le_que;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> result_org_mask_que;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> result_add_mask_que;

    // Temporary buffers
    AscendC::TBuf<AscendC::TPosition::VECCALC> start_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> end_buf; 
    
    // Temporary buffers continued
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputFloat_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> validOffset_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> vocabMask_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ones_buf_;
    
    __gm__ scalar_t *input_, *masked_input_;
    __gm__ bool *mask_out_;
    int64_t size_;
    int64_t org_vocab_start_index_, org_vocab_end_index_;
    int64_t added_vocab_start_index_, added_vocab_end_index_;
    int64_t added_offset_;

    static constexpr float MIN_ACCURACY_FP32 = 1.1754943508222875e-38;
    static constexpr float MAX_MUL_1_FP32 = 1125899906842624;
    static constexpr float MAX_MUL_2_FP32 = 67108864;
    static constexpr float NEGATIVE_ONE_FP32 = -1.0f;
};

extern "C" __global__ __aicore__ void get_masked_input_and_mask_kernel(
    __gm__ int32_t* input,
    __gm__ int32_t* masked_input,
    __gm__ bool* mask_out, 
    const int64_t org_vocab_start_index,
    const int64_t org_vocab_end_index,
    const int64_t num_org_vocab_padding,
    const int64_t added_vocab_start_index,
    const int64_t added_vocab_end_index,
    const int64_t size,
    const uint32_t loop_cnt,
    const uint32_t aiv_num)
{
    {
        GetMaskedInputAndMask<int32_t> op{};

        for (int64_t i = AscendC::GetBlockIdx(); i < loop_cnt; i += aiv_num) {
            op.Init(input + i * size/loop_cnt, 
                   masked_input + i * size/loop_cnt,
                   mask_out + i * size/loop_cnt,
                   org_vocab_start_index, org_vocab_end_index,
                   num_org_vocab_padding, added_vocab_start_index,
                   added_vocab_end_index, size/loop_cnt);
                
            op.Process();
        }
    } // op destructor called here
}

namespace vllm_ascend {

void get_masked_input_and_mask_impl(
    void* stream,
    void* input,
    void* masked_input,
    void* mask_out,
    const int64_t org_vocab_start_index,
    const int64_t org_vocab_end_index,
    const int64_t num_org_vocab_padding, 
    const int64_t added_vocab_start_index,
    const int64_t added_vocab_end_index,
    const int64_t size,
    const uint32_t loop_cnt,
    const uint32_t aiv_num)
{
    get_masked_input_and_mask_kernel<<<aiv_num, nullptr, stream>>>(
        static_cast<int32_t*>(input),
        static_cast<int32_t*>(masked_input),
        static_cast<bool*>(mask_out),
        org_vocab_start_index,
        org_vocab_end_index,
        num_org_vocab_padding,
        added_vocab_start_index,
        added_vocab_end_index,
        size,
        loop_cnt,
        aiv_num);
}

} // namespace vllm_ascend
    
