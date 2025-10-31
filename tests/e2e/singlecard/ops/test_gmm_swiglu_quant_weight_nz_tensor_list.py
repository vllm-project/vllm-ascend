import gc
import math
import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

def shape_nd_to_nz(shape):
    assert len(shape) >= 2
    batch = shape[:-2]
    a, b = shape[-2], shape[-1]
    a0, b0 = 16, 32
    return list(batch) + [math.ceil(b / b0), math.ceil(a / a0), a0, b0]

def gen_axes_for_transpose(offset, base):
    return [x for x in range(offset)] + [x + offset for x in base]

def convert_nd_to_nz(x):
    array_trans = gen_axes_for_transpose(len(x.shape) - 2, [2, 0, 1, 3])
    x_shape = shape_nd_to_nz(x.shape)
    *_, n1, m1, m0, n0 = x_shape
    return x.reshape(x_shape[:-4] + [m1, m0, n1, n0]).permute(*array_trans)

def gmm_swiglu_quant(x: torch.Tensor, weight: torch.Tensor, perChannelScale: torch.Tensor, perTokenScale: torch.Tensor, m: int):
    """
    执行量化的 GMM(通用矩阵乘法)操作,并使用 SwiGLU 激活函数。

    参数:
        x (torch.Tensor): 输入张量,形状为 (m, k)。
        weight (torch.Tensor): 权重张量,形状为 (k, n)。
        perChannelScale (torch.Tensor): 每个通道的缩放因子,形状为 (n,)。
        perTokenScale (torch.Tensor): 每个 token 的缩放因子,形状为 (m,)。
        m (int): token 的数量(x 的行数)。

    返回:
        quantOutput (torch.Tensor): 量化后的输出张量,形状为 (m, k // 2)。
        quantScaleOutput (torch.Tensor): 量化缩放因子,形状为 (m,)。
    """
    # 使用 int32 精度执行矩阵乘法
    c_temp1 = torch.matmul(x.to(torch.int32), weight.to(torch.int32))
    c_temp1 = c_temp1.to(torch.float32)  # 转换回 float32 以便进行缩放

    # 应用每个通道和每个 token 的缩放
    c_temp2 = torch.mul(c_temp1, perChannelScale)
    c_temp3 = torch.mul(c_temp2, perTokenScale.reshape(m, 1))

    # 将结果分成两部分以应用 SwiGLU 激活函数
    c_temp4, gate = c_temp3.chunk(2, dim=-1)
    c_temp5 = c_temp4 * torch.sigmoid(c_temp4)  # SwiGLU 激活
    c_temp6 = c_temp5 * gate  # 与门控值进行逐元素相乘

    # 对输出进行量化
    max = torch.max(torch.abs(c_temp6), -1).values  # 找到最大绝对值以计算缩放因子
    quantScaleOutput = 127 / max  # 计算量化缩放因子
    quantOutput = torch.round(
        c_temp6 * quantScaleOutput.reshape(m, 1)).to(torch.int8)  # 量化为 int8
    quantScaleOutput = 1 / quantScaleOutput  # 反向量化缩放因子以便后续反量化

    return quantOutput, quantScaleOutput

def process_groups(x: torch.Tensor, weight: torch.Tensor,
                   perChannelScale: torch.Tensor, perTokenScale: torch.Tensor,
                   groupList: torch.Tensor):
    """
    按组处理输入数据,并调用 GMM_Swiglu_quant 函数进行量化计算。

    参数:
        x (torch.Tensor): 输入张量,形状为 (M, K)。
        weight (torch.Tensor): 权重张量列表,每个元素的形状为 (E, K, N)。
        perChannelScale (torch.Tensor): 每个通道的缩放因子列表,每个元素的形状为 (E, N)。
        perTokenScale (torch.Tensor): 每个 token 的缩放因子,形状为 (M,)。
        groupList (list): 定义每个组的 token 数量的列表。

    返回:
        quantOutput (torch.Tensor): 量化后的输出张量,形状为 (M, N // 2)。
        quantScaleOutput (torch.Tensor): 量化缩放因子,形状为 (M,)。
    """
    print(f"x shape: {x.shape}")
    print(f"weight shape: {weight.shape}")
    print(f"perChannelScale shape: {perChannelScale.shape}")
    print(f"perTokenScale shape: {perTokenScale.shape}")
    print(f"groupList shape: {groupList.shape}")
    M, N = x.shape[0], weight.shape[2]  # 获取输入张量的形状
    quantOutput = torch.zeros(M, N // 2).to(torch.int8)  # 初始化量化输出张量
    quantScaleOutput = torch.zeros(M).to(torch.float32)  # 初始化量化缩放因子张量

    start_idx = 0  # 起始索引
    preV = 0  # 前一个组的 token 数量
    groupList = groupList.tolist()
    # 遍历 groupList,按组处理数据
    for i, v in enumerate(groupList):
        currV = v
        tempV = currV - preV  # 计算当前组的 token 数量
        preV = currV  # 更新前一个组的 token 数量
        if tempV > 0:
            # 调用 GMM_Swiglu_quant 处理当前组
            quantOutput[start_idx:start_idx + tempV], quantScaleOutput[start_idx:start_idx + tempV] = \
                gmm_swiglu_quant(x[start_idx:start_idx + tempV],
                                 weight[i],
                                 perChannelScale[i],
                                 perTokenScale[start_idx:start_idx + tempV],
                                 tempV)

        start_idx += tempV  # 更新起始索引以处理下一组
    return quantOutput, quantScaleOutput

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

    weight_nz_npu = []
    weight_scale_npu = []
    for i in range(E):
        weight_nz = convert_nd_to_nz(weight[i].clone()).npu()
        weight_nz_npu.append(weight_nz)
        weight_scale_npu.append(weight_scale[i].clone().npu())

    # x_scale (M,) - float32
    x_scale = torch.rand(M) * 0.9 + 0.1  # uniform(0.1, 1.0)
    x_scale = x_scale.to(torch.float32)

    group_list = torch.tensor([2048, 4096, 6144, 8192], dtype=torch.int64)

    output_cpu, output_scale_cpu = process_groups(x, weight, weight_scale, x_scale, group_list)
    output_npu, output_scale_npu, _ = \
        torch.ops._C_ascend.grouped_matmul_swiglu_quant_weight_nz_tensor_list(x.npu(),
                                                                              weight_nz_npu,
                                                                              weight_scale_npu,
                                                                              x_scale.npu(),
                                                                              group_list.npu())
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
