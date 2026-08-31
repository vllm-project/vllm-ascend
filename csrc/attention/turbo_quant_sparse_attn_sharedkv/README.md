# TurboQuantSparseAttnSharedkv

## 功能说明

`TurboQuantSparseAttnSharedkv` 面向 DeepSeek V4 DSA 的 shared-KV 稀疏注意力场景，将 4bit
TurboQuant 解压与 `SparseAttnSharedkv` 的 sparse-compressor 分支融合在同一个 kernel 内。
压缩 KV 不生成中间反量化张量，避免将 512 维 BF16/FP16 latent 写回 GM。

算子同时消费原始窗口 KV 和压缩稀疏 KV：原始窗口保持 BF16/FP16，压缩 KV 使用
`TurboQuantCompressLatent(output_mode=1)` 生成的 compact slot。每个 slot 固定为 258 字节：

```text
[0, 256)   512 个 4bit code，偶数维在低 nibble
[256, 258) float16 corrected_scale
```

解压过程为：

```text
code[d] = unpack(slot[d / 2])
kv_hat[d] = centroid[code[d]] * corrected_scale
```

其中 `corrected_scale = norm(latent) / norm(centroid[code])`。query、KV 和 attention output
位于同一个 signed-Hadamard 基中；调用方在 attention 前变换 query，并在输出后执行逆变换。

## 产品支持

| 产品 | 支持情况 |
| --- | --- |
| Atlas A2 推理系列 | 支持 |
| Atlas A3 推理系列 | 支持 |

## 接口约束

| 参数 | 类型 | 约束 |
| --- | --- | --- |
| `q` | FP16/BF16 | TND，末维固定 512 |
| `ori_kv` | FP16/BF16 | PA_ND，与 `q` dtype 一致，末维固定 512 |
| `cmp_kv` | UINT8 | PA_ND，末维固定 258 |
| `cmp_sparse_indices` | INT32 | sparse-compressor 选中的压缩 KV 索引 |
| `ori_block_table` / `cmp_block_table` | INT32 | PageAttention block table |
| `cu_seqlens_q` / `seqused_kv` | INT32 | TND query 累积长度与每请求有效 KV 长度 |
| `sinks` | FLOAT32 | 每个 query head 的 attention sink |
| `metadata` | INT32 | 复用 `SparseAttnSharedkvMetadata` 的调度结果 |
| `attn_out` | FP16/BF16 | shape 与 `q` 一致 |

- `kv_quant_mode` 仅支持 3，默认值为 3。
- `layout_q` 仅支持 `TND`，`layout_kv` 仅支持 `PA_ND`。
- 仅支持 sparse-compressor（SCFA）模板；`ori_kv` 和 `cmp_kv` 均须存在。
- `cmp_sparse_indices`、两份 block table、`cu_seqlens_q`、`seqused_kv`、`sinks` 和 `metadata` 均为必需输入；`ori_sparse_indices` 与 `seqused_q` 必须为空。
- `cmp_kv_stride` 以字节为单位，必须是 258 的正整数倍。
- `q`、`ori_kv` 和 `attn_out` 支持 FP16/BF16；`cmp_kv` 必须为 UINT8。
- query head 数须为 4 的倍数，KV head 数固定为 1，head dim 固定为 512。
- `cmp_sparse_indices` 最后一维支持 512 或 1024；`cmp_ratio` 支持 4 和 128，TurboQuant 4bit 调用使用 4。
- `ori_mask_mode=4`、`cmp_mask_mode=3`、`ori_win_right=0`；`ori_win_left` 必须为非负值，
  由 vLLM-Ascend 按模型的 `sliding_window - 1` 传入（默认值 127 用于兼容现有调用）。
- `sinks` 为 FP32，shape 为 `[num_query_heads]`；`metadata` 为 1024 个 INT32。
- `return_softmax_lse=false` 时第二输出为空；为 true 时输出 shape 为 `q.shape[:-1] + [1]`，dtype 为 FP32。

## 实现结构

- `op_host/`：独立 OpDef、推形和 tiling；在 host 阶段收窄 dtype、layout、slot 和模板范围。
- `op_kernel/`：SCFA cube/vector 流程；vector0 将稀疏 slot 搬入 UB、按固定码本批量解压，随后复用
  原 sparse attention 的 MM1、softmax 和 MM2 流水。
- `vllm_ascend/turboquant/`：负责 Hadamard 前后变换、TurboQuant 压缩写入和 shared-KV 调用参数编排；
  压缩写侧通过 `output_mode=1` 生成 258B compact corrected slot。
- `tests/pytest/`：不依赖 NPU 的协议数学测试。

压缩写侧由 `cann/ops-nn` 的 `TurboQuantCompressLatent` 提供。默认 `output_mode=0` 保持原
320B GLM 布局；本算子必须与 `output_mode=1` 的 258B compact corrected 布局配套使用。
