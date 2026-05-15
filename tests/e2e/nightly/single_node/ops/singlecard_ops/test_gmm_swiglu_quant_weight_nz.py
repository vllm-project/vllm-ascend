import gc

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

# enable internal format
torch_npu.npu.config.allow_internal_format = True
# enable vllm-ascend custom ops
enable_custom_op()


def gmm_swiglu_quant(x: torch.Tensor, weight: torch.Tensor,
                     perChannelScale: torch.Tensor,
                     perTokenScale: torch.Tensor, m: int, swiglu_limit: float):
    """
    Perform quantized GMM (Grouped Matrix Multiplication) operation with SwiGLU activation function.

    Parameters:
        x (torch.Tensor): Input tensor with shape (m, k).
        weight (torch.Tensor): Weight tensor with shape (k, n).
        perChannelScale (torch.Tensor): Per-channel scaling factor with shape (n,).
        perTokenScale (torch.Tensor): Per-token scaling factor with shape (m,).
        m (int): Number of tokens (rows of x).

    Returns:
        quantOutput (torch.Tensor): Quantized output tensor with shape (m, k // 2).
        quantScaleOutput (torch.Tensor): Quantization scaling factor with shape (m,).
    """
    # Perform matrix multiplication with int32 precision
    c_temp1 = torch.matmul(x.to(torch.int32), weight.to(torch.int32))
    c_temp1 = c_temp1.to(torch.float32)  # 转换回 float32 以便进行缩放

    # 应用每个通道和每个 token 的缩放
    c_temp2 = torch.mul(c_temp1, perChannelScale)
    c_temp3 = torch.mul(c_temp2, perTokenScale.reshape(m, 1))

    # 将结果分成两部分以应用 SwiGLU 激活函数
    gate, up = c_temp3.chunk(2, dim=-1)
    if swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    c_temp6 = gate * torch.sigmoid(
        gate) * up  # Element-wise multiplication with gating values

    # 对输出进行量化
    abs_max = torch.max(torch.abs(c_temp6), -1).values  # 找到最大绝对值以计算缩放因子
    quantScaleOutput = 127 / abs_max  # 计算量化缩放因子
    quantOutput = torch.round(c_temp6 * quantScaleOutput.reshape(m, 1)).to(
        torch.int8)  # 量化为 int8
    quantScaleOutput = 1 / quantScaleOutput  # 反向量化缩放因子以便后续反量化

    return quantOutput, quantScaleOutput


def process_groups(x: torch.Tensor, weight: torch.Tensor,
                   perChannelScale: torch.Tensor, perTokenScale: torch.Tensor,
                   groupList: torch.Tensor, swiglu_limit: float):
    """
    Process input data by groups and call GMM_Swiglu_quant function for quantized computation.

    Parameters:
        x (torch.Tensor): Input tensor with shape (M, K).
        weight (torch.Tensor): List of weight tensors, each with shape (E, K, N).
        perChannelScale (torch.Tensor): List of per-channel scaling factors, each with shape (E, N).
        perTokenScale (torch.Tensor): Per-token scaling factor with shape (M,).
        groupList (list): List defining the number of tokens in each group.

    Returns:
        quantOutput (torch.Tensor): Quantized output tensor with shape (M, N // 2).
        quantScaleOutput (torch.Tensor): Quantization scaling factor with shape (M,).
    """
    M, N = x.shape[0], weight.shape[2]  # Get the shape of the input tensor
    quantOutput = torch.zeros(M, N // 2).to(
        torch.int8)  # Initialize quantized output tensor
    quantScaleOutput = torch.zeros(M).to(
        torch.float32)  # Initialize quantization scaling factor tensor

    start_idx = 0  # Starting index
    preV = 0  # Number of tokens in the previous group
    groupList = groupList.tolist()
    # Iterate through groupList to process data by groups
    for i, v in enumerate(groupList):
        currV = v
        tempV = currV - preV  # Calculate number of tokens in the current group
        preV = currV  # Update number of tokens in the previous group
        if tempV > 0:
            # Call GMM_Swiglu_quant to process the current group
            quantOutput[start_idx:start_idx + tempV], quantScaleOutput[start_idx:start_idx + tempV] = \
                gmm_swiglu_quant(x[start_idx:start_idx + tempV],
                                 weight[i],
                                 perChannelScale[i],
                                 perTokenScale[start_idx:start_idx + tempV],
                                 tempV,swiglu_limit)

        start_idx += tempV  # Update starting index to process the next group
    return quantOutput, quantScaleOutput


def generate_non_decreasing_sequence(length, upper_limit):
    """
        生成一个随机非减的一维 Tensor，且最后一个值小于上限。

        参数:
            length (int): 序列的长度。
            upper_limit (int): 最后一个值的上限。

        返回:
            torch.Tensor: 生成的一维 Tensor。
        """
    # 生成随机递增序列
    random_increments = torch.randint(0, 128, (length, ))  # 随机增量，范围 0~9
    sequence = torch.cumsum(random_increments, dim=0)  # 累加生成非减序列

    # 确保最后一个值小于上限
    if sequence[-1] >= upper_limit:
        scale_factor = upper_limit / sequence[-1]  # 计算缩放因子
        sequence = (sequence * scale_factor).to(torch.int64)  # 缩放并转换为整数

    return sequence


@torch.inference_mode()
def test_gmm_swiglu_quant_weight_nz_tensor_list():
    M, K, E, N = 8192, 7168, 4, 4096

    # x (M, K) - int8
    x = torch.randint(-128, 127, (M, K), dtype=torch.int8)

    # weight (E, N, K) - int8
    weight = torch.randint(-128, 127, size=(E, K, N), dtype=torch.int8)

    # weight_scale (E, N) - float32
    weight_scale = torch.rand(E, N) * 0.9 + 0.1  # uniform(0.1, 1.0)
    weight_scale = weight_scale.to(torch.float32)
    weight_npu = torch_npu.npu_format_cast(weight.npu(), 29)
    # x_scale (M,) - float32
    x_scale = torch.rand(M) * 0.9 + 0.1  # uniform(0.1, 1.0)
    x_scale = x_scale.to(torch.float32)

    group_list = torch.tensor([2048, 4096, 6144, 8192], dtype=torch.int64)
    swiglu_limit = 1
    output_cpu, output_scale_cpu = process_groups(x, weight, weight_scale,
                                                  x_scale, group_list,
                                                  swiglu_limit)
    output_npu, output_scale_npu, _ = \
        torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz(x.npu(),
                                                                              weight_npu,
                                                                              weight_scale.npu(),
                                                                              x_scale.npu(),
                                                                              group_list.npu(),swiglu_limit=swiglu_limit)
    output_npu_valid = output_npu[:group_list[-1], :]
    output_scale_npu_valid = output_scale_npu[:group_list[-1]]

    torch.testing.assert_close(output_npu_valid.cpu(),
                               output_cpu,
                               atol=1,
                               rtol=2**-13)
    torch.testing.assert_close(output_scale_npu_valid.cpu(),
                               output_scale_cpu,
                               atol=1e-9,
                               rtol=1e-6)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
