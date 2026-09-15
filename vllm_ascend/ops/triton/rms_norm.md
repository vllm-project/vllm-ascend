# `triton_q_rms` 算子文档

## 1、算子功能介绍

`triton_q_rms` 是 vllm-ascend 在 Triton 上实现的 RMSNorm（Root Mean Square Layer Normalization）算子，文件位置：`vllm_ascend/ops/triton/rms_norm.py`。

RMSNorm 是 Transformer 模型中广泛使用的一种归一化操作，其计算流程为：

1. 对输入 `x` 的最后一维计算**均方值**：`variance = mean(x²)`；
2. 对输入做**归一化**：`output = x / sqrt(variance + ε)`。

相比 LayerNorm，RMSNorm 移除了均值减法和可学习参数（scale/bias），计算量更小，在大模型（如 LLaMA、Qwen、MiniMax 等）中被广泛采用。

本算子的 Triton kernel 行为：

- 将输入 `(batch_size, num_heads, head_dim)` 展平为 `(total_batch, head_dim)`，每个 program（Vector Core）处理一段连续的 batch 行。
- 每个 program 内通过 `BLOCK_M` 大小的 tile 对 batch 维度进行分块，逐块加载、计算 variance、归一化、写回。
- **自适应 BLOCK_M 计算**：根据 `head_dim`（即 `DIM`）、输入数据类型及元素位宽，自动估算能填入 Unified Buffer（UB）的行数 `ROW_BLOCK_SIZE`，再经 `min(batch_per_core)` 截断并取不大于它的最大 2 的幂得到 `BLOCK_M`。此机制取代了硬编码的 `ROW_BLOCK_SIZE = 16`，使小 hidden_size 获得更大的批量并行度、大 hidden_size 避免 UB 溢出。

### 限制

- `head_dim > 2048` 时抛出 `NotImplementedError`（当前 Kernel 的 UB 调度限制）。
- 仅支持 `total_batch >= 1` 的输入。

---

## 2、参数含义介绍

### 2.1 Triton Kernel：`triton_rms_kernel`

#### 运行时参数（设备侧张量/标量）

| 参数名 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `hidden_state_ptr` | `[total_batch, DIM]` 张量指针（fp16 / bf16 / fp32） | 输入张量 `q` 展开后的指针，对应的数据布局为 `[total_batch, head_dim]`（contiguous）。 |
| `hidden_state_stride_bs` | int | 输入张量的第一维 stride（即 `head_dim`，等价于 `q.stride(0)`）。 |
| `norm_output_ptr` | `[total_batch, DIM]` 张量指针，与输入同 dtype | 输出张量指针，存储归一化后的结果。 |
| `variance_epsilon` | fp32 标量 | RMSNorm 的 epsilon，防止除以零。 |
| `total_batch` | int | 展平后的总行数，即 `batch_size * num_heads`。 |

#### 编译期常量参数（`tl.constexpr`）

| 参数名 | 含义 |
| --- | --- |
| `DIM` | 输入向量的最后一维长度（即 `head_dim`），当前限制 `DIM <= 2048`。 |
| `BLOCK_M` | Batch 维度的 tile 大小，即每个 tile 并行处理的行数。由 `triton_q_rms` 函数自适应计算（详见 2.2）。 |

### 2.2 入口函数：`triton_q_rms`

#### 参数

| 参数名 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `q` | `[batch_size, num_heads, head_dim]`, fp16 / bf16 / fp32 | 输入张量，需 contiguous。 |
| `variance_epsilon` | float | RMSNorm 的 epsilon。 |

#### 自适应 `BLOCK_M` 计算流程

```python
Max_UB_Size = 1572864            # Ascend A2/A3 Unified Buffer 大小（字节）
Element_Size = element_size() * 8  # 每个元素占用的位数（如 bf16 = 16 位）

# 冗余系数：覆盖 UB 中并存的中间张量
if Element_Size == 32:
    Data_Multiplier = 5   # input(32bit) + output(32bit) + offset(32bit) + mask(32bit) + others(32bit)
elif Element_Size == 16:
    Data_Multiplier = 7   # input(16bit) + output(16bit) + offset(32bit) + mask(32bit) + others(16bit)
else:
    raise NotImplementedError  # 不支持的位宽

# 估算 UB 可容纳的行数（向下取整）
ROW_BLOCK_SIZE = int(Max_UB_Size / (dim * Element_Size * Data_Multiplier))

# 不超出每个 Core 分到的行数
raw = min(ROW_BLOCK_SIZE, batch_per_core)
BLOCK_M = 1 << (raw.bit_length() - 1)   # 不大于 raw 的最大 2 的幂
```

> **说明**：`ROW_BLOCK_SIZE` 依据元素位宽自动选择 `Data_Multiplier`——fp32 需要更少冗余（5x，因为中间张量多为 32 位原生），fp16/bf16 需要更多冗余（7x，offset/mask 等控制张量以 32 位存在）。最终 `BLOCK_M` 必须为 2 的幂以适配 Triton 向量化。

### 2.3 启动网格

```python
grid = (num_vectorcore,)
```

每个 Vector Core 负责处理 `ceil(total_batch / num_vectorcore)` 行数据。通过 `BLOCK_M` 在 Core 内部进一步分块。

---

## 3、算子使用示例

假设：

- `batch_size = 1`，`num_heads = 32`，`head_dim = 128`；
- 输入 dtype 为 `torch.bfloat16`，`eps = 1e-5`。

```python
import torch

from vllm_ascend.ops.triton.rms_norm import triton_q_rms
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

# 在 module 级别初始化 Triton device 属性（仅在测试/非推理入口时需手动调用）
init_device_properties_triton()

device = "npu"

# ---- 1. 构造输入 ----
batch_size, num_heads, head_dim = 1, 32, 128
eps = 1e-5

q = torch.randn(batch_size, num_heads, head_dim, dtype=torch.bfloat16, device=device)

# ---- 2. 调用 triton_q_rms ----
output = triton_q_rms(q, eps)

# ---- 3. 与 PyTorch 参考实现比对 ----
def rms_norm_ref(x: torch.Tensor, eps: float) -> torch.Tensor:
    x_fp32 = x.cpu().float()
    out = x_fp32 * torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    return out.to(x.dtype)

expected = rms_norm_ref(q, eps)
torch.testing.assert_close(output.float().cpu(), expected.float().cpu(), rtol=2e-2, atol=5e-2)
print("RMSNorm 精度验证通过")
```

### 多 shape 验证（推荐）

可使用以下 shape + dtype 组合对算子的精度和自适应块大小进行验证：

| shape | head_dim | dtype | 预期 BLOCK_M（A2/A3） |
|---|---|---|---|
| (100, 32, 128) | 128 | fp16/bf16/fp32 | 64 |
| (50, 16, 512) | 512 | fp16/bf16/fp32 | 16 |
| (10, 17, 2048) | 2048 | fp16/bf16/fp32 | 4 |
| (100, 64, 64) | 64 | fp16 | 128 |
| (100, 16, 256) | 256 | bf16 | 32 |
| (30, 32, 1024) | 1024 | bf16 | 8 |