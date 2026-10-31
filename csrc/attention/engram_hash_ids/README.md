# EngramHashIds

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
|<term>Ascend 950PR&950DT 系列产品</term>| √ |
|<term>Atlas A3 系列产品</term>| √ |
|<term>Atlas A2 系列产品</term>| √ |
|<term>Atlas 200I/500 A2 推理产品</term>| × |
|<term>Atlas 推理系列产品</term>| × |
|<term>Atlas 训练系列产品</term>| × |

## 功能说明

EngramHashIds 用于计算 Engram 多层、多阶 n-gram 的哈希索引，对应
[engram_hash.py](../../../vllm_ascend/ops/triton/engram_hash.py) 中 `_hash_ids_kernel` 算子的 Ascend C 实现。

对每个 token，按当前请求的位置依次选择当前 chunk、lookback 窗口或 cache 中的历史值。
当前 chunk 和已知 lookback token 先经过 `token_map` 映射；cache 已存放映射后的值。
无 cache 且历史未知时使用 `pad_id`。如果历史位置为负或历史值等于 `dead_id`，
当前及后续阶次统一使用 `pad_id`。

对每个 layer 独立计算，每个 layer 对应一组 `multipliers`、`primes` 和 `offsets`：

```text
rolling = 0
for shift in [0, MAX_NGRAM):
    rolling ^= value[shift] * multipliers[layer, shift]
    if shift > 0:
        output[token, layer, (shift - 1) * num_heads + head] =
            rolling % primes[layer, shift - 1, head] + offsets[layer, shift - 1, head]
```

乘法和异或按 64 位整数回绕；取余遵循有符号整数除法向零截断的规则。
加 offset 后保留低 32 位作为输出，不使用浮点近似或饱和转换。
无 cache 的未知历史回退值先截断为 int32，再扩展为 int64；被屏蔽位置使用完整 int64 `pad_id`。
本算子消费已有的 `req_ids`，不执行请求索引搜索，也不更新 cache。

## 参数说明

以下表格描述 PyTorch 接口。`T` 为有效 token 数，`L` 为 layer 数，`N` 为最大 n-gram 长度，
`H` 为 head 数，`R` 为请求数，`D` 为 lookback 深度。所有 Tensor 均为 ND 格式。
表中 shape 表示逻辑布局，实际缓冲区可以大于有效尺寸。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 |
| --- | --- | --- | --- |
| `input_ids` | 输入 | 当前 chunk 的 token ID，逻辑 shape `[T]`。 | INT32、INT64 |
| `token_map` | 输入 | 原始 token ID 到哈希源值的映射，shape `[vocabulary]`。 | INT32、INT64 |
| `dead_mask` | 输入 | 当前 chunk 的无效 token 标记，逻辑 shape `[T]`。 | BOOL |
| `positions` | 输入 | 每个 token 在请求内的位置，逻辑 shape `[T]`。 | INT32、INT64 |
| `block_table` | 可选输入 | cache 的逻辑块到物理块映射，逻辑 shape `[num_table_rows, max_blocks]`。 | INT32 |
| `query_start_loc` | 输入 | 请求在当前 chunk 中的 token 起点，逻辑 shape `[R + 1]`。 | INT32 |
| `req_ids` | 输入 | 每个 token 所属请求的索引，逻辑 shape `[T]`。 | INT32 |
| `multipliers` | 输入 | 每层、每个历史偏移的乘数，按 `[L, N]` 线性排列。 | INT64 |
| `primes` | 输入 | 每层、每个输出阶次及 head 的非零除数，按 `[L, N - 1, H]` 线性排列。 | INT64 |
| `offsets` | 输入 | 哈希输出的偏移，与 `primes` 排列一致。 | INT64 |
| `cache` | 可选输入 | 已映射的历史值，连续的一维 slot 缓冲区。 | INT32 |
| `lookback_token_ids` | 输入 | 请求的近期历史 token，逻辑 shape `[R, D]`；负值表示未知。 | INT32、INT64 |
| `lookback_dead_mask` | 输入 | lookback 历史的无效标记，逻辑 shape `[R, D]`。 | BOOL |
| `output` | 输出 | 调用方预分配的连续存储，逻辑 shape `[T, L, (N - 1) * H]`。 | INT32 |
| `num_tokens` | 属性 | 本次计算的有效 token 数 T。 | INT64 |
| `num_slots` | 属性 | cache 中可访问的有效 slot 数。 | INT64 |
| `pad_id` | 属性 | 历史缺失或被屏蔽时使用的值。 | INT64 |
| `input_stride`、`mask_stride`、`position_stride` | 属性 | `input_ids`、`dead_mask`、`positions` 的元素步长。 | INT64 |
| `table_stride`、`table_col_stride` | 属性 | `block_table` 的行、列元素步长。 | INT64 |
| `query_stride` | 属性 | `query_start_loc` 的元素步长。 | INT64 |
| `num_table_rows`、`max_blocks`、`cache_block_size` | 属性 | cache 映射表的有效行数、每行逻辑块数及每块 slot 数。 | INT64 |
| `MAX_NGRAM`、`num_heads`、`num_layers` | 属性 | 最大 n-gram 长度 N、head 数 H、layer 数 L。 | INT64 |
| `dead_id` | 属性 | 表示无效历史源值的标记。 | INT64 |
| `lookback_depth` | 属性 | 有效 lookback 深度 D。 | INT64 |
| `lookback_row_stride`、`lookback_col_stride` | 属性 | `lookback_token_ids` 的行、列元素步长。 | INT64 |
| `lookback_mask_row_stride`、`lookback_mask_col_stride` | 属性 | `lookback_dead_mask` 的行、列元素步长。 | INT64 |

PyTorch 接口原地写入 `output`，返回 `None`。

## 约束说明

- 所有 Tensor 必须位于同一 NPU。调用方需保证请求索引、已知 token 的词表索引及各缓冲区地址有效。
- `token_map`、`req_ids`、`multipliers`、`primes`、`offsets`、`cache` 和 `output` 必须连续。
  其余输入按显式传入的元素 stride 寻址，支持非负 stride，包括零 stride。
- `num_tokens >= 0`、`num_layers > 0`、`MAX_NGRAM >= 1`、`num_heads > 0`、`lookback_depth >= 0`。
  除数必须非零；支持负除数，并单独处理 `INT64_MIN % -1`。
- `cache=None` 时不读取 `block_table`，未知历史使用 pad 回退；使用 cache 时，必须提供非空
  `block_table`、非空 cache 和正的有效 cache 尺寸。cache 寻址包含位置、表行和 slot 截断。
- cache 位置上界目前按 64 位计算。若 Triton 的运行时标量发生 32 位乘法溢出，两者在该极端输入下可能不同。
- 输入可以使用固定容量缓冲区。`num_tokens` 等有效尺寸和 stride 必须描述本次实际数据。
  输出仅写入前 `T * L * (N - 1) * H` 个元素，其余容量不变；有效输出为空时不启动 kernel。
- 替换 Triton 调用时，保留数据参数顺序，使用普通 `torch.ops` 调用，移除
  `BLOCK_T`、`BLOCK_H`、`num_warps`，并显式传入 `num_layers`。

## 调用示例

```python
torch.ops._C_ascend.engram_hash_ids(
    input_ids, token_map, dead_mask, positions, block_table, query_start_loc,
    req_ids, multipliers, primes, offsets, cache, lookback_token_ids,
    lookback_dead_mask, output, num_tokens, num_slots, pad_id,
    input_stride=input_ids.stride(0),
    mask_stride=dead_mask.stride(0),
    position_stride=positions.stride(0),
    table_stride=block_table.stride(0) if block_table is not None else 0,
    table_col_stride=block_table.stride(1) if block_table is not None else 0,
    query_stride=query_start_loc.stride(0),
    num_table_rows=num_table_rows, max_blocks=max_blocks,
    cache_block_size=cache_block_size, MAX_NGRAM=max_ngram, num_heads=num_heads,
    dead_id=dead_id, lookback_depth=lookback_depth,
    lookback_row_stride=lookback_token_ids.stride(0),
    lookback_col_stride=lookback_token_ids.stride(1),
    lookback_mask_row_stride=lookback_dead_mask.stride(0),
    lookback_mask_col_stride=lookback_dead_mask.stride(1),
    num_layers=num_layers,
)
```

## 测试说明

正确性测试：[test_engram_hash_ids.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_engram_hash_ids.py)。
共 735 个 case，要求 Ascend C、Triton 与 CPU 参考完全一致，并检查输出保护区和只读输入。

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_engram_hash_ids.py
```

性能测试：[benchmark_engram_hash_ids.py](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/benchmark_engram_hash_ids.py)。
对比两种实现的时延和加速比。

```bash
python tests/e2e/nightly/single_node/ops/singlecard_ops/benchmark_engram_hash_ids.py
```
