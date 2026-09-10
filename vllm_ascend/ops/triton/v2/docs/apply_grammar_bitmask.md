# apply_grammar_bitmask

## 功能说明

- 算子功能：`apply_grammar_bitmask` 用于 structured output / grammar decoding 场景，
  根据压缩的 grammar bitmask 对 logits 进行原地屏蔽。bitmask 中 bit 为 `1` 表示对应
  token 允许生成，保持 logit 不变；bit 为 `0` 表示对应 token 被 grammar 禁止，将
  对应 logit 填充为 `-inf`。

- `bitmask` 使用 INT32 压缩存储，每个 INT32 表示连续 32 个 token 的允许/禁止状态。

- `logits_indices` 用于建立 grammar bitmask 行与实际 logits 行之间的映射：
  `logits_indices[i]` 表示第 `i` 行 bitmask 应作用到哪一行 logits。

- 计算公式：

  对第 `r` 行 bitmask、第 `v` 个 token：

  $$
  word = \left\lfloor \frac{v}{32} \right\rfloor,\qquad
  bit = v \bmod 32
  $$

  $$
  allowed_{r,v}
  =
  \mathbb{1}
  \left[
  bitmask_{r,word} \;\&\; (1 \ll bit) \neq 0
  \right]
  $$

  最终：

  $$
  logits_{logits\_indices[r],v}
  =
  \begin{cases}
  logits_{logits\_indices[r],v}, & allowed_{r,v}=1 \\
  -\infty, & allowed_{r,v}=0
  \end{cases}
  $$

- 算法流程：

  1. **Logical task 划分**：upstream 按
     `(num_masks, ceil(vocab_size / BLOCK_SIZE))` 构造 logical grid，每个 logical
     task 负责一行 bitmask 中一个 `BLOCK_SIZE` 宽度的 vocab block。

  2. **VectorCore 均分任务**：Ascend NPU 上根据可用 VectorCore 数量确定实际 launch
     grid，并将所有 logical task 尽可能平均分配到各 Triton program，每个 program
     的任务数最多相差 1。

  3. **Bitmask 解包**：每次加载对应 vocab block 的 packed INT32 bitmask，通过按位与
     判断每个 token 对应 bit 是否为 0。

  4. **应用掩码**：仅对 bit 为 0 的 token 写入 `-inf`；bit 为 1 的位置不写回，保持
     原始 logits 不变。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:--------|:----------------|:------|:---------|:---------|
| logits | 输入/输出 | 二维 `[num_logits, vocab_size]` logits 张量。根据 grammar bitmask 原地将禁止 token 对应位置写为 `-inf`。 | BFLOAT16 | ND |
| logits_indices | 输入 | 一维 `[num_masks]` 映射表。`logits_indices[i]` 指定第 `i` 行 bitmask 对应的 logits 行。 | INT32 | ND |
| bitmask | 输入 | 二维 `[num_masks, ceil(vocab_size / 32)]` packed grammar bitmask，每个 INT32 保存连续 32 个 token 的允许/禁止状态。 | INT32 | ND |
| vocab_size | 属性 | 实际词表大小，用于限制最后一个 vocab block 的有效范围。 | INT | - |
| BLOCK_SIZE | 属性 | 每个 logical task 负责的 vocab block 大小。当前 A2/A3 验证配置为 `8192`。 | INT | - |

## 约束说明

- `logits` 必须为二维张量，当前业务验证数据类型为 BFLOAT16。

- `logits` 最后一维 vocab 维度必须连续，即 `stride(1) == 1`。

- `logits_indices` 必须为一维 INT32 tensor，长度等于 `bitmask.shape[0]`。

- `logits_indices[i]` 必须为合法的 logits 行下标。

- `bitmask` 必须为二维 INT32 tensor，其第二维至少满足：

  $$
  bitmask.shape[1]
  \ge
  \left\lceil
  \frac{vocab\_size}{32}
  \right\rceil
  $$

- 每个 INT32 bitmask word 对应连续 32 个 token：
  bit `0` 对应第一个 token，bit `31` 对应第 32 个 token。

- bitmask bit 为 `1` 时保持对应 logit 不变；bit 为 `0` 时将对应 logit 写为 `-inf`。

- 支持 `logits_indices` 非 identity 映射，即 bitmask 行号与 logits 行号不要求相同。

- 最后一个 vocab block 不足 `BLOCK_SIZE` 时，通过 `block_offset < vocab_size` 做越界保护。

- 当前 A2/A3 验证配置中 `BLOCK_SIZE=8192`，每个 logical task 直接处理完整 8192-token
  vocab block，不再额外使用 `BLOCK_SIZE_SUB` 二次切分。

- Ascend NPU 上实际 launch grid 根据 VectorCore 数量确定：

  ```python
  num_programs = min(get_vectorcore_num(), total_tasks)
  ```

  logical task 会按 `num_programs` 均分，每个 program 的任务数最多相差 1。

- 当前实现使用 `multibuffer=False`。

- 当前 PR 的功能与性能验证范围为 Ascend A2 / A3；A5 的进一步 grid / tile 优化不在
  当前 PR 范围内。

- 当前重点业务 shape：

  ```text
  logits.shape         = [64, 151936]
  logits.dtype         = BFLOAT16
  logits_indices.shape = [64]          # packed (req_idx, position) mapping
  logits_indices.dtype = INT32
  cu_num_logits.shape  = [num_reqs + 1]
  cu_num_logits.dtype  = INT32
  bitmask.shape        = [64, 4748]
  bitmask.dtype        = INT32
  BLOCK_SIZE           = 8192
  ```

- 图模式支持情况：待补充。

## 调用示例

```python
import torch

from vllm.triton_utils import triton
from vllm_ascend.ops.triton.v2.apply_grammar_bitmask import (
    _apply_grammar_bitmask_kernel,
)

device = torch.device("npu:0")

rows = 64
num_reqs = 16
logits_per_req = 4
vocab_size = 151936
block_size = 8192
mask_stride = 8  # must exceed the largest packed position
bitmask_words = (vocab_size + 31) // 32

logits = torch.randn(
    rows,
    vocab_size,
    dtype=torch.bfloat16,
    device=device,
)

# Cumulative logit rows per request.
cu_num_logits = torch.arange(
    0,
    (num_reqs + 1) * logits_per_req,
    logits_per_req,
    dtype=torch.int32,
    device=device,
)

# Each entry packs (req_idx, position) as req_idx * mask_stride + position.
req_idx = torch.arange(rows) // logits_per_req
position = torch.arange(rows) % logits_per_req
logits_indices = (req_idx * mask_stride + position).to(torch.int32).to(device)

# bit=1 means allowed. -1 means all 32 bits are 1.
bitmask = torch.full(
    (rows, bitmask_words),
    -1,
    dtype=torch.int32,
    device=device,
)

grid = (
    rows,
    triton.cdiv(vocab_size, block_size),
)

_apply_grammar_bitmask_kernel[grid](
    logits,
    logits.stride(0),
    logits_indices,
    cu_num_logits,
    bitmask,
    bitmask.stride(0),
    vocab_size,
    MASK_STRIDE=mask_stride,
    BLOCK_SIZE=block_size,
)
```

## test ut

```bash
pytest -sv --noconftest \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_apply_grammar_bitmask_triton.py
```
