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
        input_ = input;
        masked_input_ = masked_input;
        mask_out_ = mask_out;
        org_vocab_start_index_ = org_vocab_start_index;
        org_vocab_end_index_ = org_vocab_end_index;
        //size_ = size;
        size_ = ((size + 31) / 32) * 32;
        added_offset_ = added_vocab_start_index - 
            (org_vocab_end_index - org_vocab_start_index) - 
            num_org_vocab_padding;
        added_vocab_start_index_ = added_vocab_start_index;
        added_vocab_end_index_ = added_vocab_end_index;

        inputGlobal.SetGlobalBuffer(input);
        maskedOutputGlobal.SetGlobalBuffer(masked_input); 
        maskOutGlobal.SetGlobalBuffer(mask_out);
        //AscendC::DumpTensor(inputGlobal ,0 , 32);

        pipe.InitBuffer(inQueue, 1, size * sizeof(scalar_t));
        pipe.InitBuffer(outQueue, 1, size * sizeof(scalar_t));
        pipe.InitBuffer(maskQueue, 1, size * sizeof(bool));
        pipe.InitBuffer(calc_buf_1, size_ * sizeof(float));
        pipe.InitBuffer(calc_buf_2, size_ * sizeof(float));
        pipe.InitBuffer(result_ge_que, BUFFER_NUM, size_ * sizeof(float));
        pipe.InitBuffer(result_le_que, BUFFER_NUM, size_ * sizeof(float));
        pipe.InitBuffer(result_org_mask_que, BUFFER_NUM, size_ * sizeof(float));
        pipe.InitBuffer(result_add_mask_que, BUFFER_NUM, size_ * sizeof(float));
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

    template<typename FromType, typename ToType>
    __aicore__ inline AscendC::LocalTensor<ToType> ConvertTensor(const AscendC::LocalTensor<FromType>& input) {
        return input.template ReinterpretCast<ToType>();
    }

__aicore__ inline void CompareWithValue(
    AscendC::LocalTensor<int8_t>& result,
    //AscendC::LocalTensor<float>& result,
    const AscendC::LocalTensor<float>& input,
    const AscendC::LocalTensor<float>& compare_value,  // 修改为LocalTensor类型
    bool is_greater_equal) {

    //AscendC::DumpTensor(input, 1101, 32);
    //AscendC::DumpTensor(compare_value, 1102, 32);  // 直接打印compare_value
    //AscendC::LocalTensor<float> compute_buf;
    AscendC::LocalTensor<float> compute_buf = calc_buf_1.Get<float>();
    //result = result_ge_que.AllocTensor<float>();
    
    // Max操作的操作数顺序根据比较类型调整
    if (is_greater_equal) {
        AscendC::Max(compute_buf, input, compare_value, size_);  // 直接使用compare_value
        AscendC::Sub(compute_buf, compare_value, compute_buf, size_);  // 直接使用compare_value
    } else {
        AscendC::Max(compute_buf, input, compare_value, size_);  // 直接使用compare_value
        AscendC::Sub(compute_buf, compute_buf, compare_value, size_);  // 直接使用compare_value
    }

    AscendC::Abs(compute_buf, compute_buf, size_);
    AscendC::Mins(compute_buf, compute_buf, MIN_ACCURACY_FP32, size_);
    AscendC::Muls(compute_buf, compute_buf, MAX_MUL_1_FP32, size_);
    AscendC::Muls(compute_buf, compute_buf, MAX_MUL_1_FP32, size_);
    AscendC::Muls(compute_buf, compute_buf, MAX_MUL_2_FP32, size_);
    AscendC::Adds(compute_buf, compute_buf, NEGATIVE_ONE_FP32, size_);
    AscendC::Abs(compute_buf, compute_buf, size_);
    //AscendC::DumpTensor(compute_buf, 1103, 32);

    AscendC::LocalTensor<half> compute_buf_fp16 = calc_buf_2.Get<half>();
    AscendC::Cast(compute_buf_fp16, compute_buf, AscendC::RoundMode::CAST_NONE, size_);
    AscendC::Cast(result, compute_buf_fp16, AscendC::RoundMode::CAST_NONE, size_);
    //AscendC::DumpTensor(result, 1103, 32);
}


    // 封装范围比较的函数
__aicore__ inline void ComputeRangeMask(
    AscendC::LocalTensor<int8_t>& range_mask,
    const AscendC::LocalTensor<float>& input,
    const float start_value,
    const float end_value) {
    
    // 使用独立的缓冲区来存储比较值
    AscendC::TBuf<AscendC::TPosition::VECCALC> start_buf, end_buf;
    pipe.InitBuffer(start_buf, size_ * sizeof(float));
    pipe.InitBuffer(end_buf, size_ * sizeof(float));

    // 获取tensor并初始化
    AscendC::LocalTensor<float> start_value_tensor = start_buf.Get<float>();
    AscendC::LocalTensor<float> end_value_tensor = end_buf.Get<float>();

    // 使用Duplicate来设置值
    AscendC::Duplicate(start_value_tensor, start_value, size_);
    AscendC::Duplicate(end_value_tensor, end_value, size_);
    
    AscendC::LocalTensor<int8_t> ge_result = result_ge_que.AllocTensor<int8_t>(); 
    AscendC::LocalTensor<int8_t> lt_result = result_le_que.AllocTensor<int8_t>();
    //AscendC::LocalTensor<float> ge_result = result_ge_que.AllocTensor<float>();
    //AscendC::LocalTensor<float> lt_result = result_le_que.AllocTensor<float>();

    // 计算 >= start_value
    //AscendC::DumpTensor(input, 1000, 32);
    //AscendC::DumpTensor(start_value_tensor, 1000, 32);
    CompareWithValue(ge_result, start_value_tensor, input, true);
    //AscendC::DumpTensor(ge_result, 1100, 32);

    // 计算 < end_value
    //AscendC::DumpTensor(input, 1000, 32);
    //AscendC::DumpTensor(end_value_tensor, 1000, 32);
    CompareWithValue(lt_result, input, end_value_tensor, false);
    //AscendC::DumpTensor(lt_result, 1200, 32);

    // 合并结果
    AscendC::And(range_mask,
                 //ge_result.template ReinterpretCast<uint8_t>(),
                 //lt_result.template ReinterpretCast<uint8_t>(),
                 ge_result, lt_result,
                 size_);
    AscendC::DumpTensor(range_mask, 1210, 32);
}


__aicore__ inline void Compute() {
        AscendC::LocalTensor<scalar_t> inputLocal = inQueue.DeQue<scalar_t>();
        AscendC::LocalTensor<scalar_t> maskedLocal = outQueue.AllocTensor<scalar_t>();
        AscendC::LocalTensor<int8_t> maskLocal = maskQueue.AllocTensor<int8_t>();

        // 将输入转换为float

    AscendC::TBuf<AscendC::TPosition::VECCALC> inputFloat_buf;
    pipe.InitBuffer(inputFloat_buf, size_ * sizeof(float));

    // 获取tensor并初始化
    AscendC::LocalTensor<float> inputFloat = inputFloat_buf.Get<float>();

        //AscendC::LocalTensor<float> inputFloat;
        AscendC::Cast(inputFloat, inputLocal, AscendC::RoundMode::CAST_NONE, size_);

        // 计算org_vocab范围的掩码
        // org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
        AscendC::DumpTensor(inputFloat, 900, 32);
        AscendC::LocalTensor<int8_t> orgVocabMask = result_org_mask_que.AllocTensor<int8_t>();
        ComputeRangeMask(orgVocabMask, 
                        inputFloat,
                        static_cast<float>(org_vocab_start_index_),
                        static_cast<float>(org_vocab_end_index_));
        AscendC::DumpTensor(orgVocabMask, 1200, 32);

        // 计算added_vocab范围的掩码
        // added_vocab_mask = (input_ >= added_vocab_start_index) & (input_ < added_vocab_end_index)
        AscendC::DumpTensor(inputFloat, 1300, 32);
        AscendC::LocalTensor<int8_t> addedVocabMask = result_add_mask_que.AllocTensor<int8_t>();
        ComputeRangeMask(addedVocabMask,
                        inputFloat,
                        static_cast<float>(added_vocab_start_index_),
                        static_cast<float>(added_vocab_end_index_));
        AscendC::DumpTensor(addedVocabMask, 1400, 32);

        // 计算validOffset
        //valid_offset = (org_vocab_start_index * org_vocab_mask) + (added_offset * added_vocab_mask)
    AscendC::TBuf<AscendC::TPosition::VECCALC> validOffset_buf;
    pipe.InitBuffer(validOffset_buf, size_ * sizeof(float));

    // 获取tensor并初始化
    AscendC::LocalTensor<float> validOffset = validOffset_buf.Get<float>();

    //AscendC::LocalTensor<float> validOffset;

    #if 1
    // 使用独立的缓冲区来存储比较值
    AscendC::TBuf<AscendC::TPosition::VECCALC> start_buf;
    pipe.InitBuffer(start_buf, size_ * sizeof(float));

    // 获取tensor并初始化
    AscendC::LocalTensor<float> constOrgStartIndex = start_buf.Get<float>();

    // 使用Duplicate来设置值
    AscendC::Duplicate(constOrgStartIndex, float(org_vocab_start_index_), size_);

    AscendC::LocalTensor<half> orgVocabMask_fp16;
    AscendC::LocalTensor<float> orgVocabMask_fp32;
    AscendC::Cast(orgVocabMask_fp16, orgVocabMask, AscendC::RoundMode::CAST_NONE, size_);
    AscendC::Cast(orgVocabMask_fp32, orgVocabMask_fp16, AscendC::RoundMode::CAST_NONE, size_);

        AscendC::DumpTensor(orgVocabMask, 1401, 32);
        AscendC::DumpTensor(constOrgStartIndex, 1402, 32);
        AscendC::DumpTensor(orgVocabMask_fp16, 1404, 32);
        AscendC::DumpTensor(orgVocabMask_fp32, 1405, 32);

        AscendC::Mul(validOffset, 
            constOrgStartIndex,
            orgVocabMask_fp32,
            size_);
        AscendC::DumpTensor(validOffset, 1500, 32);
   #endif
    
    AscendC::LocalTensor<float> addedOffset;
    // 使用独立的缓冲区来存储比较值
    AscendC::TBuf<AscendC::TPosition::VECCALC> start_buf_;
    pipe.InitBuffer(start_buf_, size_ * sizeof(float));

    // 获取tensor并初始化
    AscendC::LocalTensor<float> addedOffsetTensor = start_buf_.Get<float>();

    // 使用Duplicate来设置值
    AscendC::Duplicate(addedOffsetTensor, float(added_offset_), size_);

    AscendC::LocalTensor<half> addedVocabMask_fp16;
    AscendC::LocalTensor<float> addedVocabMask_fp32;
    AscendC::Cast(addedVocabMask_fp16, addedVocabMask, AscendC::RoundMode::CAST_NONE, size_);
    AscendC::Cast(addedVocabMask_fp32, addedVocabMask_fp16, AscendC::RoundMode::CAST_NONE, size_);

        AscendC::DumpTensor(addedVocabMask, 1501, 32);
        AscendC::DumpTensor(addedVocabMask_fp16, 1502, 32);
        AscendC::DumpTensor(addedVocabMask_fp32, 1503, 32);
        AscendC::DumpTensor(addedOffsetTensor, 1504, 32);

        AscendC::Mul(addedOffset, 
            addedOffsetTensor,
            addedVocabMask_fp32,
            size_);
        AscendC::DumpTensor(addedOffset, 1600, 32);

        AscendC::DumpTensor(validOffset, 1700, 32);
        AscendC::DumpTensor(addedOffset, 1800, 32);
        AscendC::Add(validOffset, validOffset, addedOffset, size_);
        AscendC::DumpTensor(validOffset, 1900, 32);

        // 计算最终掩码

    AscendC::TBuf<AscendC::TPosition::VECCALC> vocabMask_buf_;
    pipe.InitBuffer(vocabMask_buf_, size_ * sizeof(int8_t));
        
        AscendC::LocalTensor<int8_t> vocabMask  = vocabMask_buf_.Get<int8_t>();

        AscendC::Or(vocabMask,
                    //orgVocabMask.template ReinterpretCast<bool>(),
                    //addedVocabMask.template ReinterpretCast<bool>(),
                    orgVocabMask,
                    addedVocabMask,
                    size_);
        AscendC::DumpTensor(vocabMask, 2000, 32);

        // 计算最终结果
        AscendC::DumpTensor(inputFloat, 2010, 32);
        AscendC::DumpTensor(validOffset, 2020, 32);
        AscendC::Sub(inputFloat, inputFloat, validOffset, size_);
        AscendC::DumpTensor(inputFloat, 2030, 32);
        AscendC::DumpTensor(vocabMask, 2040, 32);


    AscendC::LocalTensor<half> vocabMask_fp16;
    AscendC::LocalTensor<float> vocabMask_fp32;
    AscendC::Cast(vocabMask_fp16, vocabMask, AscendC::RoundMode::CAST_NONE, size_);
    AscendC::Cast(vocabMask_fp32, vocabMask_fp16, AscendC::RoundMode::CAST_NONE, size_);
    AscendC::LocalTensor<float> inputFloat_fp32;
        AscendC::Mul(inputFloat,
            inputFloat,
            vocabMask_fp32,
            size_);
        AscendC::DumpTensor(inputFloat, 2050, 32);
        //AscendC::DumpTensor(maskedLocal, 2051, 32);

        //AscendC::LocalTensor<half> maskedLocal_int64;
        //AscendC::Cast(maskedLocal_int64, inputFloat, AscendC::RoundMode::CAST_CEIL, size_); // cast精度有问题
        //AscendC::DumpTensor(maskedLocal_int64, 2052, 32);

    AscendC::TBuf<AscendC::TPosition::VECCALC> maskedLocal_buf_;
    pipe.InitBuffer(maskedLocal_buf_, size_ * sizeof(int32_t));
    AscendC::LocalTensor<int32_t> maskedLocal_int32 = maskedLocal_buf_.Get<int32_t>();

        AscendC::Cast(maskedLocal_int32, inputFloat, AscendC::RoundMode::CAST_CEIL, size_);  // 512 直接指定元素数量
        AscendC::DumpTensor(maskedLocal_int32, 20521, 32);
        outQueue.EnQue(maskedLocal_int32);
  
// 创建并初始化全0的tensor
    AscendC::TBuf<AscendC::TPosition::VECCALC> ones_buf_;
    pipe.InitBuffer(ones_buf_, size_ * sizeof(float));
AscendC::LocalTensor<float> ones_tensor = ones_buf_.Get<float>();
AscendC::Duplicate(ones_tensor, (float)1, size_);
AscendC::LocalTensor<float> maskLocal_fp32;

        AscendC::DumpTensor(vocabMask_fp32, 2081, 32);
        AscendC::DumpTensor(ones_tensor, 2090, 32);
        AscendC::Sub(maskLocal_fp32, 
            ones_tensor,    // 全1的tensor
            vocabMask_fp32,     // 原始mask
            size_);
        AscendC::DumpTensor(maskLocal_fp32, 2100, 32);

    AscendC::LocalTensor<half> maskLocal_fp16;
    AscendC::Cast(maskLocal_fp16, maskLocal_fp32, AscendC::RoundMode::CAST_NONE, size_);
    AscendC::DumpTensor(maskLocal_fp16, 2100, 32);
    AscendC::Cast(maskLocal, maskLocal_fp16, AscendC::RoundMode::CAST_NONE, size_);
    AscendC::DumpTensor(maskLocal, 2100, 32);
        

        //outQueue.EnQue(maskedLocal);
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
    static constexpr int32_t BUFFER_NUM = 2; // tensor num for each queuea
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
    GetMaskedInputAndMask<int32_t> op{};

    for (int64_t i = AscendC::GetBlockIdx(); i < loop_cnt; i += aiv_num) {
        AscendC::printf("fmt string %s\n", "op.Init start");
        op.Init(input + i * size/loop_cnt, 
               masked_input + i * size/loop_cnt,
               mask_out + i * size/loop_cnt,
               org_vocab_start_index, org_vocab_end_index,
               num_org_vocab_padding, added_vocab_start_index,
               added_vocab_end_index, size/loop_cnt);
		AscendC::printf("fmt string %s\n", "op.Init end");
            
		AscendC::printf("fmt string %s\n", "op.Process start");
        op.Process();
		AscendC::printf("fmt string %s\n", "op.Process end");
    }
	AscendC::printf("fmt string %s\n", "op.Process end 1111");
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

