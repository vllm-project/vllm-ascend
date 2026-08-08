# apply_top_k_top_p_triton

## 功能说明

- 算子功能：`apply_top_k_top_p_triton` 基于 [Qrita](https://arxiv.org/abs/2602.01518)
  （Pivot-based Truncation and Selection）论文思想，用 Triton 实现融合的 top-k 与 top-p
  采样掩码：先按 logit 值做 top-k 截断，再对剩余 token 的 softmax 概率分布做 top-p
  截断（保留累积概率首次达到 $p$ 的最小集合），并将被过滤的位置填充为 `mask_value`
  （默认 `-inf`）。

- 计算公式：

    - Top-k（按 logit 值保留最大的 $k$ 个，含重复值处理）：

        $$
        mask_i^{(k)} = \mathbb{1}\left[logit_i > pivot_k\right]
        $$

      其中 $pivot_k$ 为第 $k$ 大 logit 对应的截断阈值（三元搜索求得，重复值按需保留）。

    - Top-p（在 top-k 保留集合上按概率截断）：

        $$
        prob_i = \frac{e^{logit_i - max}}{\sum_j e^{logit_j - max}}, \qquad
        \text{保留最小集合 } S \text{ 使 } \sum_{i \in S} prob_i \ge p
        $$

      其中 $max = \max_j logit_j$ 用于数值稳定。

- 算法流程（逐行独立处理）：

    1. **Gaussian sigma 截断**：从单个采样 block 估计均值/标准差，结合查询表得到
       sigma，计算 outlier pivot，第一遍扫描将高于该 pivot 的 logit 视为候选 outlier
       并聚集到内部 BUFFER，大幅缩小后续搜索范围。
    2. **Top-k 搜索**：对候选/全量数据做三元搜索（ternary search）求 $pivot_k$，
       单次融合扫描同时统计 `k_pivots_num`、`min_larger`、`num_min_larger`。
    3. **Top-p 搜索**：将 top-k 保留集合作 softmax 归一化后，对概率做二分搜索求
       $pivot_p$，同样单次融合扫描。
    4. **应用掩码**：按最终 pivot 生成 keep_mask（含重复 logit 的数量控制），
       被过滤位置写 `mask_value`。

- 支持三种模式：仅 top-k（`p=None`）、仅 top-p（`k=None`）、top-k + top-p 融合；
  二者均为 `None` 时是 no-op，直接返回输入。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:--------|:----------------|:------|:---------|:---------|
| logits | 输入/输出 | 二维 `[batch_size, vocab_size]` logits 张量。返回张量可能复用该输入（原地修改），对于不支持的排布（最后一维非连续）会返回新的连续张量。 | FLOAT32 | ND |
| k | 输入 | 一维 `[batch_size]`，每行 top-k 保留数量；传 `None` 表示禁用 top-k。`k >= vocab_size` 视为禁用 top-k。内部转换为 INT32。 | INT32 | ND |
| p | 输入 | 一维 `[batch_size]`，取值范围 [0, 1]；传 `None` 表示禁用 top-p。`p = 1.0` 视为禁用 top-p。内部转换为 FLOAT32。 | FLOAT32 | ND |
| mask_value | 属性 | 被过滤位置填充的值，默认 `-inf`。 | FLOAT | - |

## 约束说明

- `logits` 必须为二维且数据类型为 FLOAT32。
- `logits` 每一行的 vocab 维度要求连续（`stride(1) == 1`）；非连续输入会被内部拷贝为连续，此时返回新张量、不满足原地修改语义。
- `k` 必须为一维、长度等于 `batch_size`；取值待补充（结合 `[0, vocab_size]`，`k >= vocab_size` 时禁用 top-k）。
- `p` 必须为一维、长度等于 `batch_size`；取值待补充（结合 `[0, 1]`，`p = 1.0` 时禁用 top-p）。
- `batch_size == 0`，或 `k` 与 `p` 同时为 `None` 时，直接返回输入，不启动 kernel。
- 支持大量 `-inf` logits（如 grammar/structured-output bitmask 场景），统计与搜索过程显式排除 `-inf`，不会产生 NaN；有限值数量不足 $k$ 时保留全部有限值。
- 支持 logits 全为 `-inf` 的场景（no-op，保持 `-inf`）。
- 支持 top-k / top-p 边界与退化场景（`k=1`、`p` 极小、多行重复 logit 等）。
- 图模式支持情况：待补充。
- NPU 上 BLOCK_SIZE 相关行为见代码内注释；最大 `vocab_size` / `batch_size` 限制待补充。

## 调用示例

```python
import torch

from vllm_ascend.ops.triton.v2.sample.apply_top_k_top_p_triton import (
    apply_top_k_top_p_triton,
)

device = "npu"
logits = torch.randn(4, 32000, dtype=torch.float32, device=device)
k = torch.tensor([50, 100, 1, 32000], dtype=torch.int32, device=device)
p = torch.tensor([0.9, 0.9, 0.5, 1.0], dtype=torch.float32, device=device)

# 融合 top-k + top-p;logits 原地被修改并返回
out = apply_top_k_top_p_triton(logits, k, p)

# 仅 top-k(禁用 top-p)
out = apply_top_k_top_p_triton(logits.clone(), k, None)

# 仅 top-p(禁用 top-k)
out = apply_top_k_top_p_triton(logits.clone(), None, p)
```

## test ut

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_apply_top_k_top_p_triton.py #--noconftest
```
