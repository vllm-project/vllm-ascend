# precopy_mamba_align_fused_kernel

## 功能说明

- 算子功能：在 Mamba 前向计算前，将跨 block 请求的旧运行状态复制到目标 block；`src_col < 0` 或 `src_col == dst_col` 时跳过。
- 计算公式：
    - SD 卷积：`state[dst, :W-bias, ...] = state[src, bias:, ...]`。
    - DS 卷积：`state[dst, row, :W-bias] = state[src, row, bias:]`。
    - Temporal：`state[dst] = state[block_table[src_col+bias]]`。
- 算法流程（逐请求、逐状态独立处理）：
  1. 从三维 grid 获取 `batch_idx`、`state_idx` 和 `tile_idx`。
  2. 可选地通过 `idx_mapping_ptr` 获取请求下标，并查询源、目标物理 block。
  3. 卷积状态仅由 tile 0 复制；DS 布局逐 dim row 复制，SD 布局连续复制。
  4. Temporal 状态由 `TEMPORAL_TILES` 个 CTA 分段复制，tail 由 tile 0 处理。
- 支持模式：SD/DS 卷积布局、temporal 状态、V1/V2 请求索引、单 tile/多 tile temporal 复制。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:--------|:----------------|:------|:---------|:---------|
| mamba_state_idx_ptr | 输入 | 目标 block-table 列。 | INT32 | 1D |
| src_col_ptr | 输入 | 源 block-table 列；负数表示跳过。 | INT32 | 1D |
| token_bias_ptr | 输入 | 卷积切片或 temporal 源列偏移。 | INT32 | 1D |
| block_table_ptrs_ptr | 输入 | 各 group 的 block-table 地址。 | INT64 | 1D |
| block_table_stride_req | 输入 | block table 请求行步长。 | INT64 | Scalar |
| state_base_addrs_ptr | 输入/输出 | 状态基地址，目标 block 原地更新。 | INT64 | 1D |
| state_block_strides_ptr | 输入 | 状态 block 字节步长。 | INT64 | 1D |
| state_elem_sizes_ptr | 输入 | 状态元素字节数。 | INT32 | 1D |
| state_inner_sizes_ptr | 输入 | temporal block 元素数或卷积 inner size。 | INT64 | 1D |
| state_conv_widths_ptr | 输入 | 卷积宽度；0 表示 temporal。 | INT32 | 1D |
| state_group_indices_ptr | 输入 | state 到 block-table group 的映射。 | INT32 | 1D |
| state_dim_row_count_ptr | 输入 | DS 卷积 dim row 数。 | INT32 | 1D |
| state_dim_row_stride_ptr | 输入 | DS 卷积 dim row 字节步长。 | INT64 | 1D |
| idx_mapping_ptr | 输入 | 可选的 batch 到请求槽位映射。 | INT32/None | 1D/- |
| num_reqs | 输入 | 有效请求数。 | INT | Scalar |
| COPY_BLOCK_SIZE | 属性 | 向量化复制宽度。 | tl.constexpr | Scalar |
| CONV_STATE_DIM_FIRST | 属性 | 是否使用 DS 布局。 | tl.constexpr | Scalar |
| HAS_IDX_MAPPING | 属性 | 是否使用请求映射。 | tl.constexpr | Scalar |
| TEMPORAL_TILES | 属性 | temporal 复制 CTA 数量。 | tl.constexpr | Scalar |

## 约束说明

- grid 必须为 `(num_reqs, num_layers * num_state_types, TEMPORAL_TILES)`。
- `COPY_BLOCK_SIZE`、`TEMPORAL_TILES` 必须为正整数编译期常量。
- 有效请求的列下标、偏移和 block ID 必须在合法范围内；卷积要求 `0 <= token_bias < conv_width`。
- DS 布局必须提供正确的 row count 和 row stride；各状态元数据长度必须一致。
- Temporal 主体按 `uint64` 复制，剩余 0～7 字节按 `uint8` 复制。
- 支持 Ascend NPU Triton JIT；图模式要求 grid 和编译期属性固定。

## 调用示例

```python
grid = (num_reqs, num_layers * num_state_types, temporal_tiles)
precopy_mamba_align_fused_kernel[grid](
    dst_col, src_col, token_bias, block_table_ptrs, block_table_stride,
    state_base_addrs, state_block_strides, state_elem_sizes, state_inner_sizes,
    state_conv_widths, state_group_indices, state_dim_row_count,
    state_dim_row_stride, idx_mapping, num_reqs, COPY_BLOCK_SIZE=128,
    CONV_STATE_DIM_FIRST=False, HAS_IDX_MAPPING=True,
    TEMPORAL_TILES=temporal_tiles,
)
```

## test ut

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_precopy.py
```
