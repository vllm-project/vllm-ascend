import gc

import numpy as np
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def x_int8_to_x_int4(x: torch.Tensor):
    m, k = x.shape
    x_high_4bit = torch.floor(x.to(torch.float16) // 16).to(torch.int8)
    x_low_4bit = (
        torch.bitwise_and(x.view(torch.int16), 0x0f0f).view(torch.int8) - 8)
    x_int4 = torch.empty((2 * m, k), dtype=torch.int8)
    x_int4[::2, :] = x_high_4bit
    x_int4[1::2, :] = x_low_4bit
    return x_int4


def custom_mm(x: torch.Tensor, weight: torch.Tensor,
              weight_scale: torch.Tensor, m: int):
    """
    Performing Quantized GMM (General Matrix Multiplication) Operation
    Parameters:
        x (torch.Tensor): Input tensor with shape (m, k).
        weight (torch.Tensor): Weight tensor with shape (k, n).
        weight_scale (torch.Tensor): Scaling factor for each channel.
          - In perGroup scenario: Shape is (k_group_num, n). Note: When k_group_num == 1, it is a perChannel scenario.
          - In perChannel scenario: Shape is (n).
        m (int): Number of tokens (number of rows in x).
    Returns:
        mm_out(fp16): Result of MatMul + perGroup or perChannel dequantization.
    """
    # Perform matrix multiplication with int32 precision
    k, n = weight.shape
    mm_out = torch.zeros((m, n), dtype=torch.float16)
    # perGroup scenario
    if len(weight_scale.shape) == 2 and weight_scale.shape[0] != 1:
        k_group = weight_scale.shape[0]
        per_group_ele = k // k_group
        x_grouped = x.view(-1, k_group, per_group_ele).transpose(0, 1)
        weight_grouped = weight.view(k_group, per_group_ele, n)

        c_temp = torch.bmm(x_grouped.to(torch.int32),
                           weight_grouped.to(torch.int32)).to(torch.float16)
        for k_idx in range(k_group):
            mm_out += (c_temp[k_idx] *
                       weight_scale[k_idx].view(1, -1).to(torch.float16)).to(
                           torch.float16)
    # perChannel scenario
    elif len(weight_scale.shape) == 1 or (len(weight_scale.shape) == 2
                                          and weight_scale.shape[0] == 1):
        c_temp = torch.matmul(x.to(torch.int32),
                              weight.to(torch.int32)).to(torch.float32)
        mm_out = c_temp * weight_scale.view(1, -1).to(torch.float16)
    return mm_out.to(torch.float32)


def gmm_swiglu_quant_golden_a8_w4(x: torch.Tensor, weight: torch.Tensor,
                                  weight_scale: torch.Tensor,
                                  per_token_scale: torch.Tensor,
                                  bias: torch.Tensor,
                                  group_list: torch.Tensor):
    """
    Process the input data by group and call the GMM_Swiglu_quant function for quantization computation.
    Parameters:
        x (torch.Tensor): Input tensor with shape (M, K), type INT8.
        weight (torch.Tensor): List of weight tensors, each with shape (E, K, N), data type INT8 but data range INT4, representing INT4 values.
        weight_scale (torch.Tensor): Scaling factor for each channel.
          - In perGroup scenario: shape (E, k_group_num, N).
          - In perChannel scenario: shape (E, N).
        per_token_scale (torch.Tensor): Scaling factor for each token, shape (M, ).
        bias: torch.Tensor,
        group_list (list): List defining the number of tokens in each group.
    Returns:
        quant_output (torch.Tensor): Quantized output tensor with shape (M, N // 2).
        quant_scale_output (torch.Tensor): Quantization scaling factor, shape (M, ).
    """
    M, N = x.shape[0], weight.shape[2]
    quant_output = torch.zeros(M, N // 2).to(torch.int8)
    quant_scale_output = torch.zeros(M).to(torch.float32)
    # Preprocessing X_INT8 -> X_INT4
    x_int4 = x_int8_to_x_int4(x)
    start_idx = 0
    # Number of tokens in the previous group
    pre_v = 0
    group_list = group_list.tolist()
    # Traverse group_list and process data by group
    for i, v in enumerate(group_list):
        curr_v = v
        # Calculate the number of tokens in the current group " * 2 " because 1 row of Int8--> 2 rows of Int4
        temp_v = int((curr_v - pre_v) * 2)
        # Update the number of tokens in the previous group
        pre_v = curr_v
        if (temp_v > 0):
            mm_out = custom_mm(x_int4[int(start_idx):int(start_idx + temp_v)],
                               weight[i], weight_scale[i], temp_v)
            mm_num_concat = ((mm_out[::2] * 16 + mm_out[1::2]) +
                             bias[i].view(1, -1))
            per_token_quant = mm_num_concat * per_token_scale[start_idx // 2:(
                start_idx + temp_v) // 2].view(-1, 1)
            swiglu, gate = per_token_quant.chunk(2, dim=-1)
            temp = swiglu * torch.sigmoid(swiglu)
            temp = temp * gate
            max_value = torch.max(torch.abs(temp), dim=-1).values
            quant_scale_output_temp = 127 / max_value
            quant_output[start_idx // 2:(start_idx + temp_v) //
                         2] = torch.round(temp *
                                          quant_scale_output_temp.reshape(
                                              temp_v // 2, 1)).to(torch.int8)
            quant_scale_output[start_idx // 2:(start_idx + temp_v) //
                               2] = 1 / quant_scale_output_temp
        start_idx += temp_v
    return quant_output, quant_scale_output


def generate_non_decreasing_sequence(length, upper_limit):
    # Generate random increasing sequence
    random_increments = torch.randint(0, 128, (length, ))
    sequence = torch.cumsum(random_increments, dim=0)

    # Make sure the last value is less than the upper limit
    if sequence[-1] >= upper_limit:
        scale_factor = upper_limit / sequence[-1]
        sequence = (sequence * scale_factor).to(torch.int64)
    return sequence


@torch.inference_mode()
def test_grouped_matmul_swiglu_quant_kernel():
    E = 16
    M = 512
    K = 7168
    N = 4096
    torch.npu.config.allow_internal_format = True
    x = torch.randint(-5, 5, (M, K), dtype=torch.int8).npu()
    weight_ori = torch.randint(-5, 5, (E, K, N), dtype=torch.int8)
    weight_nz = torch_npu.npu_format_cast(weight_ori.npu().to(torch.float32),
                                          29)
    pack_weight = torch_npu.npu_quantize(weight_nz,
                                         torch.tensor([1.], device='npu'),
                                         None, torch.quint4x2, -1, False)

    weight_scale = torch.randn(E, 1, N)
    scale_np = weight_scale.cpu().numpy()
    scale_np.dtype = np.uint32
    scale_uint64_tensor = torch.from_numpy(scale_np.astype(np.int64)).npu()
    pertoken_scale = torch.randn(M).to(torch.float32).npu()
    group_list = generate_non_decreasing_sequence(E, M).npu()
    bias = torch.zeros((E, N), dtype=torch.float32,
                       device="npu").uniform_(-5, 5)

    output_golden, output_scale_golden = gmm_swiglu_quant_golden_a8_w4(
        x.cpu(), weight_ori, weight_scale, pertoken_scale.cpu(), bias.cpu(),
        group_list.cpu())

    output, output_scale, _ = torch.ops._C_ascend.grouped_matmul_swiglu_quant(
        x=x,
        weight=pack_weight,
        bias=bias,
        group_list=group_list,
        weight_scale=scale_uint64_tensor,
        x_scale=pertoken_scale)
    torch.testing.assert_close(output_golden, output.cpu(), atol=1, rtol=0.005)
    torch.testing.assert_close(output_scale_golden,
                               output_scale.cpu(),
                               atol=1,
                               rtol=0.005)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
