# `_triton_rope` / `_triton_rope_siso` / `_triton_rope_fp8` 算子文档

## 1、算子功能介绍

文件位置：`vllm_ascend/ops/triton/rope.py`

Rotary Position Embedding（RoPE）是 Transformer 模型中广泛使用的一种位置编码方式，通过对 Query 和 Key 张量进行旋转操作注入位置信息。本文件实现了三个 Triton kernel，分别对应三种不同场景：

### 1.1 `_triton_rope` — 分离 Q/K RoPE

处理 `(q, k)` 双输入场景，q 和 k 各有独立张量和独立的 head 数（q 可能包含 GQA/MQA 的额外 head）。同时计算 q 和 k 的 RoPE，结果写回原张量（in-place）。支持 **NeoX 风格**（前后半分割）和**交错风格**（GPT-J，偶数/奇数下标）两种旋转模式。

### 1.2 `_triton_rope_siso` — 单输入单输出 RoPE（SISO）

处理 `qk` 单输入场景（q 和 k 已融合为同一张量，如 DeepSeek 系列等部分 MoE 模型中 QKV 融合后的 RoPE 分支）。读入 qk 整块到 UB，经 `extension.extract_slice` 切分左右半/交错对，算完用 `insert_slice` 合并后单次写回 `output_ptr`。

### 1.3 `_triton_rope_fp8` — FP8 输出 RoPE

输入 q/k 为 bf16/fp16，应用 NeoX 风格 RoPE 后，结果经**定标限幅**直接写为 `float8_e4m3fn` 格式。支持 `rope_dim < head_dim`（即 pass-through 维度）的处理，pass 部分同样做 FP8 限幅输出。专为 MiniMax-M3 等需要 FP8 KV Cache 的模型设计。

### 三种 kernel 的 RoPE 核心计算（旋转公式）

```
NeoX 风格:          交错风格（GPT-J）:
x1 = x[..., :d/2]   x1 = x[..., 0::2]
x2 = x[..., d/2:]   x2 = x[..., 1::2]
o1 = x1*cos - x2*sin  (相同)
o2 = x2*cos + x1*sin  (相同)
```

---

## 2、参数含义介绍

### 2.1 `_triton_rope`

#### 运行时参数

| 参数名 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `q_ptr` | `[num_tokens, n_qh, hd]` 张量指针 | 输入 q 张量，in-place 写回结果 |
| `q_row_stride` | int | q 第一个维度的 stride，即 `n_qh * hd` |
| `k_ptr` | `[num_tokens, n_kh, hd]` 张量指针 | 输入 k 张量，in-place 写回结果 |
| `k_row_stride` | int | k 第一个维度的 stride，即 `n_kh * hd` |
| `cos_ptr` / `cos_row_stride` | `[num_tokens, rope_dim/2]` | 预选好的 cos 张量（USE_COS_SIN=False 时） |
| `sin_ptr` / `sin_row_stride` | `[num_tokens, rope_dim/2]` | 预选好的 sin 张量（USE_COS_SIN=False 时） |
| `cos_sin_ptr` / `cos_sin_row_stride` | `[max_pos, rope_dim]` | cos+sin 拼接缓存，前半 cos 后半 sin（USE_COS_SIN=True 时） |
| `pos_ptr` | `[num_tokens]`, int64 | 每个 token 的位置 ID |
| `num_tokens` | int | 当前 batch 的 token 数 |

#### 编译期常量

| 参数名 | 含义 |
| --- | --- |
| `n_qh` | q 的头数 |
| `n_kh` | k 的 head 数（GQA/MQA 时 <= n_qh） |
| `hd` | head_dim（head 维度） |
| `rope_dim` | RoPE 应用的维度长度，需 ≤ `hd` |
| `pad_rope_dim` | `next_power_of_2(rope_dim)`，用于 Triton 向量化对齐 |
| `BLOCK_SIZE_HEAD` | head 维度的 tile 大小，防 UB 溢出 |
| `IS_NEOX_STYLE` | True=NeoX 风格，False=交错风格 |
| `USE_COS_SIN` | True=用一体的 `cos_sin_cache` 查表，False=分别使用 `cos`/`sin` |

#### 启动网格

```python
grid = (min(num_tokens, num_vectorcore),)
```

### 2.2 `_triton_rope_siso`

#### 运行时参数

| 参数名 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `qk_ptr` | `[num_tokens, n_h, hd]` 张量指针 | 输入 qk 张量 |
| `qk_row_stride` | int | qk 第一维 stride |
| `output_ptr` | `[num_tokens, n_h, hd]` 张量指针 | 输出张量（可与 `qk_ptr` 相同实现 in-place，也可不同实现 out-of-place） |
| `cos_sin_ptr` / `cos_sin_row_stride` / `pos_ptr` / `num_tokens` | 同上 | 同上 |

#### 编译期常量

| 参数名 | 含义 |
| --- | --- |
| `n_h` | head 数 |
| `hd` | head_dim |
| `rope_dim` | RoPE 维度长度 |
| `pad_n_h` | `next_power_of_2(n_h)`，Triton 对齐 |
| `pad_rope_dim` | `next_power_of_2(rope_dim)`，Triton 对齐 |
| `BLOCK_SIZE` | 等于 `pad_n_h`，batch tile 大小 |
| `IS_NEOX_STYLE` | RoPE 风格切换 |
| `USE_COS_SIN` | cos/sin 输入模式切换 |

#### 启动网格

```python
grid = (min(num_tokens, num_vectorcore),)
```

### 2.3 `_triton_rope_fp8`

#### 运行时参数

| 参数名 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `q_ptr` | `[num_tokens, n_qh, hd]`, bf16/fp16 | 输入 q 张量 |
| `q_out_ptr` | `[num_tokens, n_qh, hd]`, fp8 | FP8 输出 q |
| `k_ptr` | `[num_tokens, n_kh, hd]`, bf16/fp16 | 输入 k 张量 |
| `k_out_ptr` | `[num_tokens, n_kh, hd]`, fp8 | FP8 输出 k |
| `cos_sin_ptr` / `cos_sin_row_stride` | `[max_pos, rope_dim]` | cos+sin 拼接缓存 |
| `pos_ptr` | `[num_tokens]`, int64 | 位置 ID |
| `num_tokens` | int | token 数 |

#### 编译期常量

| 参数名 | 含义 |
| --- | --- |
| `n_qh` / `n_kh` | Q/K 头数 |
| `hd` | head_dim |
| `rope_dim` | RoPE 应用维度 |
| `pad_half` | `next_power_of_2(rope_dim // 2)` |
| `pad_pass` | `next_power_of_2(pass_dim)`，pass 维度 padding |
| `pass_dim` | `head_dim - rope_dim`，不做 RoPE 的直通维度大小 |
| `BLOCK_QH` / `BLOCK_KH` | Q/K 的 head tile 大小 |
| `FP8_MAX` | FP8 E4M3 最大值（固定 448.0） |

#### 启动网格

```python
grid = (min(num_tokens, max(vector_cores * 8, 256)),)
```

---

## 3、算子使用示例

### 3.1 `rope_forward_triton` — 分离 Q/K RoPE

假设：`num_tokens=1, n_qh=32, n_kv_head=8, head_dim=128, rope_dim=128`（GQA 场景）。

```python
import torch

from vllm_ascend.ops.triton.rope import rope_forward_triton

device = "npu"

# ---- 1. 构造输入 ----
num_tokens, n_q_head, n_kv_head, head_dim, rope_dim = 1, 32, 8, 128, 128
q = torch.randn(num_tokens, n_q_head, head_dim, dtype=torch.bfloat16, device=device)
k = torch.randn(num_tokens, n_kv_head, head_dim, dtype=torch.bfloat16, device=device)

# 预选的 cos/sin（每 token 一组）
cos = torch.randn(num_tokens, rope_dim // 2, dtype=torch.bfloat16, device=device)
sin = torch.randn(num_tokens, rope_dim // 2, dtype=torch.bfloat16, device=device)

# ---- 2. 调用 ----
q_out, k_out = rope_forward_triton(
    q, k,
    cos=cos, sin=sin, rope_dim=rope_dim,
    is_neox_style=True,
)
# q_out 和 k_out 已被 in-place 改写为 RoPE 后的结果
print(q_out.shape, k_out.shape)
```

### 3.2 `rope_forward_triton_siso` — 单输入 RoPE

```python
from vllm_ascend.ops.triton.rope import rope_forward_triton_siso

# 构造 qk 融合输入 (num_tokens, n_head, head_dim)
qk = torch.randn(1, 32, 128, dtype=torch.bfloat16, device=device)

# 方式 A：使用 cos_sin_cache（有 position 索引）
positions = torch.tensor([0], dtype=torch.int64, device=device)
cos_sin_cache = torch.randn(2048, rope_dim, dtype=torch.bfloat16, device=device)

qk_out = rope_forward_triton_siso(
    qk,
    cos_sin_cache=cos_sin_cache,
    positions=positions,
    rope_dim=128,
    is_neox_style=True,
)

# 方式 B：使用预选的 cos/sin
cos = torch.randn(1, rope_dim // 2, dtype=torch.bfloat16, device=device)
sin = torch.randn(1, rope_dim // 2, dtype=torch.bfloat16, device=device)
qk_out = rope_forward_triton_siso(
    qk, cos=cos, sin=sin, rope_dim=rope_dim, is_neox_style=False,
)
```

### 3.3 `_rope_forward_triton_fp8` — FP8 输出 RoPE

```python
from vllm_ascend.ops.triton.rope import rope_forward_triton

num_tokens, n_q_head, n_kv_head, head_dim, rope_dim = 1, 16, 16, 128, 128
positions = torch.tensor([0], dtype=torch.int64, device=device)
cos_sin_cache = torch.randn(2048, rope_dim, dtype=torch.bfloat16, device=device)

q = torch.randn(num_tokens, n_q_head, head_dim, dtype=torch.bfloat16, device=device)
k = torch.randn(num_tokens, n_kv_head, head_dim, dtype=torch.bfloat16, device=device)

# 通过 out_dtype 触发 FP8 路径
q_out, k_out = rope_forward_triton(
    q, k,
    cos_sin_cache=cos_sin_cache,
    positions=positions,
    rope_dim=head_dim,
    is_neox_style=True,
    out_dtype=torch.float8_e4m3fn,
)

print(q_out.dtype)   # torch.float8_e4m3fn
print(q_out.shape)   # (1, 16, 128)
```

### RoPE 风格说明

| 风格 | IS_NEOX_STYLE | 分割方式 | 示例模型 |
|---|---|---|---|
| NeoX | `True` | `x[..., :d/2]` 与 `x[..., d/2:]` | LLaMA, Qwen, DeepSeek |
| 交错 | `False` | `x[..., 0::2]` 与 `x[..., 1::2]` | GPT-J, MiniMax-M3 |

### Kernel 选择指南

| 场景 | 使用函数 | Kernel |
|---|---|---|
| Q/K 分开，bf16/fp16 输出 | `rope_forward_triton()` | `_triton_rope` |
| QKV 融合为单张量 | `rope_forward_triton_siso()` | `_triton_rope_siso` |
| FP8 E4M3 输出（如 MiniMax-M3） | `rope_forward_triton(..., out_dtype=torch.float8_e4m3fn)` | `_triton_rope_fp8` |