# AttnResFwd

## 产品支持情况

| 产品 | 本仓库默认构建 |
| --- | :---: |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | 未纳入默认构建 |
| Atlas 200I/500 A2 推理产品 | × |

产品范围以 `csrc/build_aclnn.sh` 的默认算子列表为准。算子定义包含 A2 注册配置，但本 PR
未将其加入 A2 默认构建列表；A3 使用 arch22 内核，A5 使用 arch35 内核。

## 功能说明

- 算子功能：将 Kimi K3 的有效历史 block residual 与当前 prefix residual 进行可学习的
  softmax 加权融合。对每个候选向量计算 RMS 归一化后的投影分数，沿 block 维做 softmax，
  最后加权求和原始向量。
- 计算公式：

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

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| `prefix_sum` | 输入 | 当前累计残差，shape 为 `[T, H]` | BFLOAT16 | ND |
| `block_residual` | 输入 | 有效历史 block，shape 为 `[T, N, H]` | BFLOAT16 | ND |
| `proj_weight` | 输入 | 分数投影权重，shape 为 `[1, H]` | BFLOAT16 | ND |
| `norm_weight` | 输入 | RMSNorm 权重，shape 为 `[H]` | BFLOAT16 | ND |
| `norm_eps` | 属性 | RMSNorm 的 epsilon，必须大于 0；PyTorch 接口必传 | float | - |
| `hidden_states` | 输出 | 加权后的残差，shape 为 `[T, H]` | BFLOAT16 | ND |
| `need_backward` | 属性 | ACLNN 接口是否保存前向中间量，默认 false | bool | - |
| `inv_rms` | 可选输出 | 每个候选向量的逆 RMS，shape 为 `[T, N + 1]` | FLOAT | ND |
| `probs` | 可选输出 | 每个候选向量的混合权重，shape 为 `[T, N + 1]` | FLOAT | ND |

`need_backward=true` 时，ACLNN 接口要求 `inv_rms` 和 `probs` 非空，候选行顺序与公式一致。
当前 PyTorch 包装固定设置 `need_backward=false`，只返回 `hidden_states`。

## 约束说明

- 四个输入张量必须位于同一 NPU，dtype 均为 BF16，shape 满足上表的对应关系。
- 模型调用前将输入转为 contiguous，并仅传入 `block_residual[:, :num_valid_blocks, :]`，
  排除预分配但尚未写入的 block。
- 模型接入在 `num_valid_blocks <= 0` 时直接返回 `prefix_sum`，不启动算子。
- 算子用于前向推理，不提供 PyTorch autograd backward。`need_backward` 仅控制前向中间量输出。
- 调用前须安装匹配的 CANN、自定义 ACLNN 算子包及 `vllm_ascend_C` 扩展，并完成算子注册。

## 调用示例

| 调用方式 | 接口/用例 | 说明 |
| --- | --- | --- |
| PyTorch 接口 | `torch.ops._C_ascend.attn_res_fwd` | 与其他已注册自定义算子相同，通过 torch.ops 调用 |
| ACLNN 接口 | [aclnn_attn_res_fwd.h](op_host/op_api/aclnn_attn_res_fwd.h) | 先调用 GetWorkspaceSize 构建 executor，再在指定 stream 上执行 |
| 单算子精度测试 | [test_attn_res_fwd.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_attn_res_fwd.py) | BF16 输出与 FP32 参考计算比较 |

以下示例在已配置 CANN 和自定义算子包环境的 NPU 进程中执行。独立脚本显式导入扩展以完成注册；
模型运行时复用框架的扩展注册，通过相同的 `torch.ops._C_ascend.attn_res_fwd` 入口调用。

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

单算子用例在仓库根目录运行，覆盖不同 token 数、block 数、hidden size 和 epsilon：

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
