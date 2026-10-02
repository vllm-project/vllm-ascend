# triton_q_rms

## 功能说明

- 算子功能：`triton_q_rms` 用 Triton 实现不带权重的 RMS Norm，主要用于 DSA
  Q RMS norm 路径。输入 `q` 的逻辑形状为 `[bs, head_num, dim]`，算子将前两维
  展平为 `total_batch = bs * head_num`，对每一行沿最后一维做 RMS 归一化，并返回与
  输入相同 shape 和 dtype 的新张量。

- 计算公式（逐行独立处理）：

    $$
    variance_b = \frac{1}{DIM}\sum_{d=0}^{DIM-1} x_{b,d}^2
    $$

    $$
    y_{b,d} = x_{b,d} \cdot \frac{1}{\sqrt{variance_b + \epsilon}}
    $$

  其中 $b \in [0, total\_batch)$，$d \in [0, DIM)$，$\epsilon$ 对应
  `variance_epsilon`。

- 算法流程：

    1. 将输入 `q` 从 `[bs, head_num, dim]` reshape 为 `[total_batch, dim]`。
    2. 读取当前 NPU 的 `num_vectorcore`，以 `(num_vectorcore,)` 作为 kernel grid。
    3. 每个 program 根据 `core_id` 和 `core_num` 切分若干行，并按 `BLOCK_M` 分块处理。
    4. 对每行加载 `DIM` 个元素，计算平方均值 `variance`。
    5. 使用 `rsqrt(variance + variance_epsilon)` 得到缩放因子并写入输出。
    6. 将输出 reshape 回 `[bs, head_num, dim]`。

- `BLOCK_M` 选择：

    $$
    batch\_per\_core = cdiv(total\_batch, num\_vectorcore)
    $$

    $$
    BLOCK\_M = 2^{\lfloor \log_2(\min(16, batch\_per\_core)) \rfloor}
    $$

  该值用于控制单次循环处理的行数，最大为 16。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:--------|:----------------|:------|:---------|:---------|
| q | 输入 | 三维 `[bs, head_num, dim]` Q 张量。前两维会被展平为 batch 行，最后一维做 RMS 归一化。 | FLOAT16 / BFLOAT16 / FLOAT32 | ND |
| variance_epsilon | 属性 | 加到 variance 上的 epsilon，用于数值稳定。 | FLOAT | - |
| return | 输出 | 与 `q` shape 和 dtype 相同的 RMS Norm 输出张量。 | 同 `q` | ND |

## 约束说明

- `q` 必须为三维张量，逻辑 shape 为 `[bs, head_num, dim]`。
- `dim <= 2048`；当 `dim > 2048` 时会抛出 `NotImplementedError`。
- `q` 需要能通过 `q.view(total_batch, dim)` 展平；通常要求输入为连续张量或具备兼容
  view 的内存布局。
- 输出通过 `torch.empty_like(q.view(total_batch, dim))` 分配，不复用输入存储，不是原地
  修改。
- kernel grid 使用当前设备的 `num_vectorcore`，每个 program 处理一段 batch 行。
- `q.stride(0)` 会作为展平后行跨度传入 kernel，最后一维按 `DIM` 连续访问。
- 当前算子不包含 RMSNorm weight 或 bias，仅执行不带权重的 RMS 归一化。
- 支持 dtype 以测试覆盖为准：FLOAT16、BFLOAT16、FLOAT32。
- `total_batch = bs * head_num` 应大于 0；空 batch 场景未单独处理。
- 图模式支持情况：待补充。

## 调用示例

```python
import torch

from vllm_ascend.ops.triton.rms_norm import triton_q_rms
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

device = "npu"
init_device_properties_triton()

bs = 2
head_num = 8
dim = 512
variance_epsilon = 1e-5

q = torch.randn(bs, head_num, dim, dtype=torch.float16, device=device)
out = triton_q_rms(q, variance_epsilon)

assert out.shape == q.shape
assert out.dtype == q.dtype
```

## test ut

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rms_norm.py #--noconftest
```
