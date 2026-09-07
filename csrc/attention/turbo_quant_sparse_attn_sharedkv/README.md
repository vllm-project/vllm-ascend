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
| `ori_kv` | FP16/BF16 | PA_BSND/PA_BNSD，与 `q` dtype 一致，末维固定 512 |
| `cmp_kv` | UINT8 | PA_BSND/PA_BNSD，末维固定 258 |
| `cmp_sparse_indices` | INT32 | sparse-compressor 选中的压缩 KV 索引 |
| `ori_block_table` / `cmp_block_table` | INT32 | PageAttention block table |
| `cu_seqlens_q` / `seqused_kv` | INT32 | TND query 累积长度与每请求有效 KV 长度 |
| `sinks` | FLOAT32 | 每个 query head 的 attention sink |
| `metadata` | INT32 | 复用 `SparseAttnSharedkvMetadata` 的调度结果 |
| `attn_out` | FP16/BF16 | shape 与 `q` 一致 |

- `kv_quant_mode` 仅支持 3，默认值为 3。
- `layout_q` 仅支持 `TND`；`layout_kv` 支持 `PA_BSND`（默认）和 `PA_BNSD`，旧名称
  `PA_ND` 作为 `PA_BSND` 的兼容别名保留。
- 仅支持 sparse-compressor（SCFA）模板；`ori_kv` 和 `cmp_kv` 均须存在。
- `cmp_sparse_indices`、两份 block table、`cu_seqlens_q`、`seqused_kv`、`sinks` 和 `metadata` 均为必需输入；`ori_sparse_indices` 与 `seqused_q` 必须为空。
- `ori_kv_stride` 以元素为单位，`cmp_kv_stride` 以字节为单位；二者可大于紧凑 block stride，
  但必须至少覆盖一个完整物理 block。
- `q`、`ori_kv` 和 `attn_out` 支持 FP16/BF16；`cmp_kv` 必须为 UINT8。
- query head 数须为 4 的倍数，KV head 数固定为 1，head dim 固定为 512。
- PageAttention block size 与原 `SparseAttnSharedkv` 一致：范围 `[16, 1024]` 且须按 16 对齐。
- cache 首轴可非连续；launcher 将 tensor 的 `stride(0)` 传给 kernel，stride 必须覆盖一个完整 block。
- TND 中单请求 `qs=0` 直接跳过；`kvs=0` 时跳过 attention 计算并将对应输出清零。
- `cmp_sparse_indices` 最后一维支持 512 或 1024。
- Host 保留原实现对 `cmp_ratio=4/128` 的支持；当前 TurboQuant 4bit 调用使用 4。
- `ori_mask_mode=4`、`cmp_mask_mode=3`、`ori_win_right=0`；`ori_win_left` 必须为非负值，
  由 vLLM-Ascend 按模型的 `sliding_window - 1` 传入（默认值 127 仅用于兼容现有调用）。
- 对 `qs>1`（MTP）按每个 query token 独立应用 causal sliding-window mask，不复用 tile 尾行的边界。
- `sinks` 为 FP32，shape 为 `[num_query_heads]`；`metadata` 为 1024 个 INT32。
- `return_softmax_lse=false` 时第二输出 shape 为 `[0]`；为 true 时 shape 为
  `q.shape[:-1] + [1]`，dtype 为 FP32。

## 实现结构

- `op_host/`：独立 OpDef、推形和 tiling；在 host 阶段收窄 dtype、layout、slot 和模板范围。
- `op_kernel/`：SCFA cube/vector 流程；vector0 将稀疏 slot 搬入 UB、按固定码本批量解压，随后复用
  原 sparse attention 的 MM1、softmax 和 MM2 流水。BF16 解压使用 256 项 byte-LUT，每次 Gather
  同时得到一个 packed byte 对应的两个 BF16 质心，再以 FP32 乘 FP16 corrected scale 并舍入回
  BF16；FP16 保留逐 4bit code 的 float 码本路径。BF16 查表批次由 4 行扩大到 16 行。
- `torch_extension/cann_ops_transformer/ops/`：PyTorch schema、Meta 推导和 ACLNN
  launcher；launcher 根据 cache tensor 的 `stride(0)` 生成 Host ABI 的 stride 属性。
- `tests/pytest/`：保留原有、不依赖 NPU 的 TurboQuant 数学协议辅助测试。
- `tests/ut/op_host/`：InferShape/InferDataType 与 arch22 tiling 正反向 UT。
- `examples/`：按 GLM 算子模式内置 float64 golden，通过 `cann_ops_transformer` 直调算子，
  执行 NPU 精度、非法参数和性能测试。

压缩写侧由 `cann/ops-nn` 的 `TurboQuantCompressLatent` 提供。默认 `output_mode=0` 保持原
320B GLM 布局；本算子必须与 `output_mode=1` 的 258B compact corrected 布局配套使用。

## 测试与性能

TQ4 c4 cache 的理论 payload 压缩比为 `512 * 2 / 258 = 3.97x`。该数值不等于整个 KV 池
或整模型 HBM 收益；端到端还包含未压缩 SWA cache、indexer cache、block table、workspace
及权重。性能报告应分别列出算子时延、KV pool 容量和端到端吞吐/并发，不混为一个指标。
