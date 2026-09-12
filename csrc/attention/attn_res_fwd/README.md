# AttnResFwd

## 功能

`AttnResFwd` 实现 Kimi K3 的 learned attention residual mixture：对历史 block residual 和当前
`prefix_sum` 分别计算 RMS 归一化后的投影分数，再沿 block 维做 softmax，用所得权重混合原始向量。
这是前向算子，当前 PyTorch 接口用于推理，不提供 autograd backward。

## 计算公式

令 `T` 为 token 数、`N` 为有效历史 block 数、`H` 为 hidden size，`B = N + 1`。
第 `N` 个候选向量是当前 prefix，其余为历史 block：

```text
v[t, n, h] = block_residual[t, n, h], 0 <= n < N
v[t, N, h] = prefix_sum[t, h]

inv_rms[t, n] = rsqrt(mean_h(v[t, n, h]^2) + norm_eps)
score[t, n] = sum_h(v[t, n, h] * inv_rms[t, n]
                    * norm_weight[h] * proj_weight[0, h])
probs[t, :] = softmax(score[t, :])
hidden_states[t, h] = sum_n(probs[t, n] * v[t, n, h])
```

中间计算使用 FP32，最终输出转换为 BF16。混合的是原始 `v`，不是 RMS 归一化后的向量。

## PyTorch 接口

```python
hidden_states = torch.ops._C_ascend.attn_res_fwd(
    prefix_sum, block_residual, proj_weight, norm_weight, norm_eps
)
```

| 参数 | Shape | Dtype / 类型 | 说明 |
| --- | --- | --- | --- |
| `prefix_sum` | `[T, H]` | BF16 | 当前累计残差 |
| `block_residual` | `[T, N, H]` | BF16 | 本次参与混合的历史 block |
| `proj_weight` | `[1, H]` | BF16 | 分数投影权重 |
| `norm_weight` | `[H]` | BF16 | RMSNorm 权重 |
| `norm_eps` | 标量 | float | 必传，必须大于 0 |
| 返回 `hidden_states` | `[T, H]` | BF16 | 加权后的残差向量 |

输入为 ND 布局，四个张量必须位于同一 NPU。模型接入在调用前将输入转为 contiguous。
PyTorch 注册入口是 `_C_ascend.attn_res_fwd`，需要先构建并加载 `vllm_ascend_C` 扩展以及对应的
自定义 ACLNN 算子包；普通 `import torch` 不会注册这个算子。

模型中的 `_apply_ascend_attn_res` 将 `block_residual[:, :num_valid_blocks, :]` 传给算子，排除
预分配但尚未写入的 block。`num_valid_blocks <= 0` 时直接返回 `prefix_sum`，不启动算子。
非空 block 路径直接调用原生算子，没有设备类型判断、算子存在性判断或 Triton/eager 回退。

## ACLNN 接口

接口声明见 [aclnn_attn_res_fwd.h](op_host/op_api/aclnn_attn_res_fwd.h)：

1. `aclnnAttnResFwdGetWorkspaceSize` 检查参数、构建 executor，并返回 workspace 大小。
2. `aclnnAttnResFwd` 在给定 stream 上执行 executor。

底层接口还提供 `needBackward` 参数。为 `true` 时，需要提供 FP32 的 `invRms` 和 `probs`
输出，shape 均为 `[T, N + 1]`，行序与公式中的候选向量一致。这些是供后续反向使用的前向
中间量，不表示该接口执行反向计算。当前 PyTorch 包装固定传 `needBackward=false`，只返回
`hidden_states`。

## 内核路径

- A3 使用 `arch22`，A5 使用 `arch35` 和 `attn_res_fwd_apt` 入口。
- tiling 根据 hidden size、block 数和 UB 容量选择 resident 或 reload 路径。
- resident 路径将候选行保留在 UB，计算分数后复用这些行做加权输出。
- reload 路径先计算分数，再重新读取候选行做加权输出。
- resident 的 UB Copy 按每次 256 字节分段执行。BF16 对应每次最多 128 个元素，不能把整行
  hidden size 作为单次 Copy 的 mask 且只执行一个 repeat，否则行尾未被正确复制。

## 调用示例

在已配置 CANN 和自定义算子包环境、并选定空闲 NPU 的进程中执行：

```python
import torch
import torch_npu
import vllm_ascend.vllm_ascend_C  # 注册自定义算子

prefix_sum = torch.randn(7, 256, device="npu", dtype=torch.bfloat16)
block_residual = torch.randn(7, 3, 256, device="npu", dtype=torch.bfloat16)
proj_weight = torch.randn(1, 256, device="npu", dtype=torch.bfloat16)
norm_weight = torch.ones(256, device="npu", dtype=torch.bfloat16)
output = torch.ops._C_ascend.attn_res_fwd(
    prefix_sum, block_residual, proj_weight, norm_weight, 1e-5
)
assert output.shape == (7, 256)
```

## 测试与验证范围

数值用例：[test_attn_res_fwd.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_attn_res_fwd.py)。
测试 BF16 输入，对比 FP32 参考计算再转 BF16 的输出；`rtol=0.01`、`atol=0.01`。

| `T` | `N` | `H` |
| --- | --- | --- |
| 1 | 1 | 128 |
| 7 | 3 | 256 |
| 32 | 8 | 4096 |
| 3 | 64 | 4096 |
| 2 | 1 | 7168 |

每组分别测试 `norm_eps=1e-5` 和 `1e-6`，共 10 项。用例覆盖超过单次 Copy 长度的行，
避免遗漏 resident 路径的行尾拷贝问题。

在仓库根目录、已构建扩展及算子包的 NPU 环境中运行：

```bash
python - <<'PY'
import torch_npu
import vllm_ascend.vllm_ascend_C
import pytest

raise SystemExit(pytest.main([
    "-v",
    "tests/e2e/nightly/single_node/ops/singlecard_ops/test_attn_res_fwd.py",
]))
PY
```

CPU 接入测试位于 [test_kimi_k3_adapter.py](../../../tests/ut/models/test_kimi_k3_adapter.py)，
通过 mock 检查有效 block 切片、contiguous 参数和零 block 返回，不在 CPU 上执行 NPU 内核。

2026-09-12：A5 上通过独立构建的 ACLNN 包和使用同一 C++ 接口的测试绑定验证，10/10 通过；
A3 尚未实机验证。该结果不等同于完整模型端到端验证。
