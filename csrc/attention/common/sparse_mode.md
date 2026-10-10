# 共享 KV 稀疏注意力的 mask 模式

本文说明仓库内 [SparseAttnSharedkv](../sparse_attn_sharedkv/README.md) 和
[KvQuantSparseAttnSharedkv](../kv_quant_sparse_attn_sharedkv/README.md) 的
`ori_mask_mode`、`cmp_mask_mode`，分别控制原始 KV 和压缩 KV 的可见范围。

## 支持的取值

| 参数 | 支持值 | 模式 | 含义 |
| --- | --- | --- | --- |
| `ori_mask_mode` | `4` | `band` | 在 query 对应的原始 token 位置附近，按 `ori_win_left`、`ori_win_right` 限定滑动窗口。 |
| `cmp_mask_mode` | `3` | `rightDownCausal` | query 与原始 KV 序列右对齐，按 `cmp_ratio` 计算压缩 KV 的因果边界。 |

两个算子的 tiling 校验均要求 `ori_mask_mode=4`、`cmp_mask_mode=3`。
其他注意力算子的 `sparse_mode` 取值不代表这两个算子也支持相同取值。
窗口大小、压缩率、数据类型和布局的支持范围，以各自 README 的约束说明为准。

## 原始 KV：band 窗口

对于一个 batch，令 `Q` 为有效 query 长度，`K` 为有效原始 KV 长度，
`i` 为从 `0` 开始的 query 行号。query 与 KV 右对齐，因此 query 对应的
原始 token 位置为 `p = K - Q + i`，最后一行对应位置 `K - 1`。

原始 KV 的可见 token 索引 `j` 满足：

```text
max(0, p - ori_win_left) <= j <= min(K - 1, p + ori_win_right)
```

两端均包含在窗口内。默认 `ori_win_left=127`、`ori_win_right=0`，
表示最多访问当前 token 和之前的 127 个 token，共 128 个 token；序列开头不足
128 个 token 时，窗口裁剪到有效 KV 范围。例如 `p=130` 时，可见索引为 `3..130`。

## 压缩 KV：rightDownCausal

令 `r = cmp_ratio`，`c` 为从 `0` 开始的压缩 KV token 索引。
其对应的压缩边界为原始 token 位置 `(c+1)*r-1`；只有该边界不超过 `p` 时，
这个压缩 token 才可见：

```text
0 <= c < floor((p + 1) / r)
```

边界按原始 KV 的位置计算，不能直接使用压缩后的 KV 长度与 query 长度做右对齐。
设置 `cmp_sparse_indices` 时，实际访问还受稀疏索引选择限制，上式不表示必须访问全部历史压缩 KV。

以下示例均为单 token decode，即 `Q=1`，使用默认窗口和 `cmp_ratio=4`，且不做稀疏选择：

| 原始 KV 长度 `K` | query 位置 `p` | 原始 KV 可见索引 | 压缩 KV 可见索引 |
| --- | --- | --- | --- |
| `7` | `6` | `0..6` | `0` |
| `8` | `7` | `0..7` | `0..1` |
| `131` | `130` | `3..130` | `0..31` |
| `132` | `131` | `4..131` | `0..32` |

例如 `K=7` 时，压缩 token `1` 的压缩边界为原始 token `7`，超过当前 query 位置 `6`，
因此不能访问；`K=8` 时 query 位置为 `7`，压缩 token `1` 才可见。

## 实现参考

- [SparseAttnSharedkv 参数校验](../sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_tiling.cpp)
- [KvQuantSparseAttnSharedkv 参数校验](../kv_quant_sparse_attn_sharedkv/op_host/kv_quant_sparse_attn_sharedkv_check_single_para.cpp)
- [SparseAttnSharedkv 窗口和压缩边界计算](../sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_scfa_kernel.h)
- [KvQuantSparseAttnSharedkv 压缩 KV 边界计算](../kv_quant_sparse_attn_sharedkv/op_kernel/arch35/kv_quant_sparse_attn_sharedkv_scfa_block_vector.h)
