# fill_logprob_token_ids

## 功能说明

- 算子功能：`fill_logprob_token_ids` 在 Ascend NPU 上构建
  logprob token IDs 矩阵及其有效性掩码，供通用的 MRV2 `compute_topk_scores` 在
  “部分请求指定了自定义 `logprob_token_ids`”（即 `max_per_req_token_ids > 0`）
  的路径下使用。它与上游 vLLM 的同名 kernel 功能一致，但针对 Ascend NPU
  修复了一个动态地址指针相关的问题（详见下文“与上游实现的差异”）。

- 输出矩阵结构：逻辑形状均为 `[batch_size, 1 + num_cols]`，其中
  `num_cols = max(num_logprobs, max_per_req_token_ids)`：

    - **第 0 列**：恒为本次采样的 token id（`sampled_token_ids[b]`），且有效性恒为
      `True`。
    - **第 1.. 列**：若该请求（经 `expanded_idx_mapping` 映射到 `req_state_idx`）
      拥有自定义 logprob token（`num_custom > 0`），则用自定义 token 覆盖 top-k
      列；否则填充该行的 top-k indices。被填充的位置才在掩码中置 `True`。

- 计算公式（逐行独立处理，`b` 为 batch 行索引）：

    - 第 0 列（恒写、恒有效）：

        $$
        out\_token\_ids[b, 0] = sampled\_token\_ids[b],
        \qquad out\_valid\_mask[b, 0] = 1
        $$

    - 请求状态映射：

        $$
        req\_state\_idx = expanded\_idx\_mapping[b],
        \qquad num\_custom = num\_per\_req\_token\_ids[req\_state\_idx]
        $$

    - 第 `1 + col` 列的有效列与数据来源（`col \in [0, PADDED\_COLS)`）：

        - 若 `num_custom > 0`（自定义 token 优先，覆盖 top-k）：

            $$
            valid = col < num\_custom,
            \qquad token = per\_req\_token\_ids[req\_state\_idx, col]
            $$

        - 否则（回退到 top-k，`NUM_TOPK == 0` 时为 no-op）：

            $$
            valid = col < NUM\_TOPK,
            \qquad token = topk\_indices[b, col]
            $$

    - 写入：

        $$
        out\_token\_ids[b, 1 + col] = token \cdot \mathbb{1}[valid],
        \qquad out\_valid\_mask[b, 1 + col] = \mathbb{1}[valid]
        $$

    其中 `PADDED_COLS = next_power_of_2(num_cols)` 是总扫描宽度，每次最多处理 `min(PADDED_COLS, COL_BLOCK_SIZE)` 列，超出
    `num_cols` 的 padding 列因 `valid = False` 不会被写入（见“约束说明”）。

- 算法流程（单次 pass、逐行并行，`tl.program_id(0)` 即 batch 行索引）：

    1. 加载并写入第 0 列（采样 token，恒有效）。
    2. 通过 `expanded_idx_mapping` 解析该行对应的请求状态索引，读取其自定义
       token 数量 `num_custom`。
    3. 按最多 4096 列分块循环，根据 `num_custom` 是否大于 0 决定数据来源（自定义 token 或 top-k indices），
       并据此确定有效列掩码 `valid`。
    4. 以掩码方式将 token（转为 INT64）写入 `[b, 1 : 1 + num_cols]`，同时将有效
       位置对应的 `out_valid_mask` 写为 1。

## 与上游实现的差异

- 上游 vLLM 版本在 `if / else` 分支外统一计算 `src` 与 `valid`，然后在分支外
  执行一次 `tl.load(src + col, ...)`。
- vLLM Ascend 版本将 `tl.load` 移入 `if / else` 两个分支内部各自执行，以规避
  Ascend NPU 上动态地址指针（`src` 需在运行时才能确定）导致的编译/执行问题。
  逻辑结果与上游完全一致，仅实现方式不同（对应源码中的注释
  “fix dynamic addr ptr by placing load inside the if-else block”）。

- 候选列采用分块循环限制 UB 使用。A5 上的长候选测试中，8192 列分块仍会触发 UB 溢出，
  4096 列通过了 A3 和 A5 的测试，因此默认 `COL_BLOCK_SIZE = 4096`。
- Python 入口负责分配输出、计算 grid 和选择 `PADDED_COLS`。底层 kernel 保留上游 patch 所用的启动参数。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:--------|:----------------|:------|:---------|:---------|
| out_token_ids_ptr | 输出 | 二维 `[batch_size, 1 + num_cols]` 输出 token IDs：第 0 列为采样 token，第 1.. 列为自定义 token 或 top-k indices，未填充位置保持输入初值（通常为 0）。 | INT64 | ND |
| out_token_ids_stride | 属性 | `out_token_ids` 的 stride(0)，即行跨度。 | INT | - |
| out_valid_mask_ptr | 输出 | 二维 `[batch_size, 1 + num_cols]` 有效性掩码，与 `out_token_ids` 逐位对应；第 0 列恒为 True，其余有效列才为 True。 | BOOL | ND |
| out_valid_mask_stride | 属性 | `out_valid_mask` 的 stride(0)。 | INT | - |
| sampled_token_ids_ptr | 输入 | 一维 `[batch_size]`，本次采样得到的 token id。 | INT64 | ND |
| topk_indices_ptr | 输入 | 二维 `[batch_size, NUM_TOPK]`，各行的 top-k token id；当 `NUM_TOPK == 0` 时不读取数据、仅作为指针占位。生产中由 `logits.topk` 结果转为 INT32 后传入。 | INT32 | ND |
| topk_indices_stride | 属性 | `topk_indices` 的 stride(0)。 | INT | - |
| expanded_idx_mapping_ptr | 输入 | 一维 `[batch_size]`，将 batch 行映射到请求状态索引 `req_state_idx`（用于索引 `num_per_req_token_ids` 与 `per_req_token_ids`）。 | INT32 | ND |
| num_per_req_token_ids_ptr | 输入 | 一维 `[max_num_reqs]`，每个请求的自定义 logprob token 数量。 | INT32 | ND |
| per_req_token_ids_ptr | 输入 | 二维 `[max_num_reqs, MAX_LOGPROB_TOKEN_IDS]`，每个请求的自定义 logprob token ids（左对齐存储）。 | INT32 | ND |
| per_req_token_ids_stride | 属性 | `per_req_token_ids` 的 stride(0)。 | INT | - |
| NUM_TOPK | 属性 | 常量，top-k 列数，通常等于 `num_logprobs`。 | constexpr INT | - |
| PADDED_COLS | 属性 | 常量，向量化扫描宽度，取值 `next_power_of_2(num_cols)`，其中 `num_cols = max(num_logprobs, max_per_req_token_ids)`。 | constexpr INT | - |
| COL_BLOCK_SIZE | 编译期属性 | 每次处理的候选列数上限，默认 4096；实际宽度不超过 PADDED_COLS。 | INT | - |

## 约束说明

- Python 入口接受非空 batch，且至少有一列自定义或默认候选容量。`topk_indices` 为 `None`
  表示无默认 top-k 候选。
  所有输入应位于同一 NPU，向量连续，矩阵列跨度为 1。
- `max_per_req_token_ids` 由调用者提供，必须覆盖当前 batch 所引用请求的自定义候选数量，
  各请求的数量不能超过 `per_req_token_ids` 的列数。所有有效 ID 都应落在词表内。
- Python 入口分配并返回新的 INT64 IDs 和 BOOL mask，未填充位置分别为 0 和 `False`，输入不发生原地修改。
  底层 kernel 仅写入调用者预先分配的输出缓冲区。
- `out_token_ids` 必须为二维 INT64，第二维至少为 `1 + num_cols`；其 dtype 与
  `sampled_token_ids` 保持一致（上游通过 `sampled_token_ids.new_zeros` 分配，即
  INT64）。
- `out_valid_mask` 必须为二维 BOOL，shape 与 `out_token_ids` 一致。
- `sampled_token_ids` 必须为一维 INT64，长度等于 `batch_size`。
- `topk_indices` 必须为二维 INT32，shape 为 `[batch_size, NUM_TOPK]`。
- `expanded_idx_mapping` 必须为一维 INT32，长度等于 `batch_size`，且每个元素
  必须是 `[0, max_num_reqs)` 内的合法请求索引。
- `num_per_req_token_ids` 必须为一维 INT32，长度等于 `max_num_reqs`。
- `per_req_token_ids` 必须为二维 INT32，shape 为
  `[max_num_reqs, MAX_LOGPROB_TOKEN_IDS]`，其中 `MAX_LOGPROB_TOKEN_IDS = 128`
  （与 `vllm.sampling_params.MAX_LOGPROB_TOKEN_IDS` 一致）。
- `PADDED_COLS` 必须为 2 的幂且大于等于 `num_cols`（用 `next_power_of_2` 保证）；
  实际写入列数由 `valid` 掩码限制在 `num_cols` 以内，越过的 padding 列不会写入，
  不会越界。
- 需满足 `num_custom <= num_cols` 与 `NUM_TOPK <= num_cols`；由
  `num_cols = max(num_logprobs, max_per_req_token_ids)` 天然保证。
- 每个 program 只处理一个 batch 行，grid 尺寸应为 `(batch_size,)`。
- 该 kernel 不处理 logits 数值，只做索引搬运与掩码写入，无浮点精度问题，对比应
  逐位相等（`torch.equal`）。

## JIT 特化

- `NUM_TOPK` 为编译期值，以便在无默认候选时消除对 `topk_indices` 的读取。
  不同的 top-k 请求值会产生不同特化，即使它们落在相同的 `PADDED_COLS` 桶内。
- `PADDED_COLS` 按 2 的幂分桶，决定遍历上界。`COL_BLOCK_SIZE` 在生产路径固定为 4096，
  `min(PADDED_COLS, COL_BLOCK_SIZE)` 决定单次载入的向量宽度。
- grid 随 batch 大小变化。实际张量类型、行跨度和对齐仍可能影响 Triton 特化。

## 调用示例

```python
import torch

from vllm_ascend.ops.triton.sample.fill_logprob_token_idx import fill_logprob_token_ids

device = "npu"
batch_size = 4
num_reqs = 4
max_per_req_token_ids = 3
num_logprobs = 5
MAX_LOGPROB_TOKEN_IDS = 128

sampled_token_ids = torch.randint(0, 1000, (batch_size,), dtype=torch.int64, device=device)
topk_indices = torch.randint(0, 1000, (batch_size, num_logprobs), dtype=torch.int32, device=device)

# batch 行 -> 请求状态索引（请求 0 拥有自定义 token）
expanded_idx_mapping = torch.arange(batch_size, dtype=torch.int32, device=device) % num_reqs
num_per_req_token_ids = torch.zeros(num_reqs, dtype=torch.int32, device=device)
per_req_token_ids = torch.zeros(num_reqs, MAX_LOGPROB_TOKEN_IDS, dtype=torch.int32, device=device)

# 请求 0 指定 3 个自定义 logprob token
num_per_req_token_ids[0] = 3
per_req_token_ids[0, 0] = 100
per_req_token_ids[0, 1] = 200
per_req_token_ids[0, 2] = 300

out_token_ids, out_valid_mask = fill_logprob_token_ids(
    sampled_token_ids,
    topk_indices,
    expanded_idx_mapping,
    num_per_req_token_ids,
    per_req_token_ids,
    max_per_req_token_ids,
)
torch.npu.synchronize()
```

## 测试用例

PyTorch CPU 参考逐元素比较 token IDs 与有效性掩码。内核测试允许 A2、A3 和 A5，
大候选生产路径验证覆盖 A3 和 A5。`test_compute_topk_scores.py` 通过 MRV2 入口覆盖自定义候选、
默认与自定义请求混合、完整词表候选以及 logits 和 logprob 两种模式。

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_fill_logprob_token_ids_kernel.py \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compute_topk_scores.py
```
