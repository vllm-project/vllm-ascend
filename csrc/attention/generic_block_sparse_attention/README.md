# GenericBlockSparseAttention

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                        |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √     |
| <term>Atlas 200I/500 A2推理系列产品</term>                    |    ×     |
| <term>Atlas 推理系列产品</term>                                |    ×     |
| <term>Atlas 训练系列产品</term>                                |    ×     |

## 功能说明

- 算子功能：`GenericBlockSparseAttention`（GBSA）按外部传入的稀疏块索引，从 **Paged KV Cache** 中选取逻辑 KV 块，执行 FlashAttention 风格的块稀疏注意力计算。任务切分依赖前置 AICPU 算子 `GenericBlockSparseAttentionMetadata` 生成的 `metadata`。

- 计算公式：

  $$
  O = \mathrm{softmax}(Q \cdot K_{\mathrm{sparse}}^{T} \cdot \mathrm{scale}) \cdot V_{\mathrm{sparse}}
  $$

  其中 $K_{\mathrm{sparse}}$ / $V_{\mathrm{sparse}}$ 由 `sparseBlockIdx`（逻辑块索引）经 `blockTable`（逻辑块 → 物理页）映射后，从 `key` / `value` cache 中 gather。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| query | 输入 | Query | FLOAT16、BFLOAT16、FLOAT8_E4M3FN | ND |
| key | 输入 | Paged Key cache | 同 query | ND |
| value | 输入 | Paged Value cache | 同 query | ND |
| sparseBlockIdx | 输入 | 每个 Q 块选中的 KV 逻辑块索引 | INT32 | ND |
| sparseBlockCount | 输入 | 每个 Q 块实际选中的 KV 块数 | INT32 | ND |
| metadata | 输入 | Metadata 算子输出的调度表，长度固定 1024 | INT32 | ND |
| attenMaskOptional | 可选输入 | Attention mask（当前常规路径不使用） | - | ND |
| q/k/vDequantScaleOptional、pQuantScaleOptional | 可选输入 | FP8 全量化相关 scale（`quantType=5` 时使用） | FLOAT | ND |
| cuSeqLengthsQOptional | 可选输入 | TND 下各 batch Query 存储长度前缀和，shape `(B+1,)` | INT64 | ND |
| cuSeqLengthsKvOptional | 可选输入 | 各 batch KV 存储长度前缀和，shape `(B+1,)` | INT64 | ND |
| sequsedQOptional | 可选输入 | 各 batch Query 实际有效长度，shape `(B,)` | INT32 | ND |
| sequsedKvOptional | 可选输入 | 各 batch KV 实际有效长度，shape `(B,)` | INT32 | ND |
| blockTableOptional | 可选输入 | 逻辑块 → 物理页映射，shape `[B, maxBlocksPerBatch]` | INT32 | ND |
| blockShape | 属性 | `[blockShapeX, blockShapeY]`，当前仅支持 `[1, 128]` | ListInt | - |
| isPackedGQA | 属性 | 是否 packed GQA；当前仅支持 `1` | INT | - |
| layoutQ | 属性 | Query 布局；当前常规路径仅支持 `"TND"` | STRING | - |
| layoutKv | 属性 | KV 布局；当前常规路径仅支持 `"PAGED_BBND"` | STRING | - |
| scaleValue | 属性 | Softmax 前缩放系数；为 0 时按 `1/sqrt(D)` | FLOAT | - |
| maskType | 属性 | Mask 类型；当前常规路径仅支持 `1`（因果） | INT | - |
| quantType | 属性 | `0`=非量化；`5`=FP8 全量化（需 Q/K/V 为 FLOAT8_E4M3FN） | INT | - |
| dstTypeMax | 属性 | 量化相关预留属性 | FLOAT | - |
| softmaxPrecision | 属性 | Softmax 精度：`0`=fp32 SM；`1`=低精度/半精度 SM | INT | - |
| winLeft / winRight | 属性 | 滑窗参数（`maskType!=2` 时保持 `-1`） | INT | - |
| returnSoftmaxlse | 属性 | 是否返回 softmax LSE；`0`/`1`；FP8 路径不支持 `1` | INT | - |
| attentionOut | 输出 | 注意力输出，shape 与 query 一致 | 同 query（FP8 时可为 FLOAT16/BFLOAT16） | ND |
| softmaxLseOptional | 可选输出 | Softmax log-sum-exp；TND 下为 `[T, N, 1]`，FLOAT | FLOAT | ND |

## 约束说明

### 常规路径（当前已支持）

- `layoutQ="TND"`，`layoutKv="PAGED_BBND"`，`maskType=1`，`blockShape=[1, 128]`，`headDim=128`，`isPackedGQA=1`。
- Query shape：`[T, Nq, D]`；Key/Value shape：`[numBlocks, blockSize, Nkv, D]`（`blockSize` 由 KV shape 解析）。
- TND + packed GQA：
  - `sparseBlockIdx`：`[Nkv, totalQBlocks, topK]`
  - `sparseBlockCount`：`[Nkv, totalQBlocks]`
  - `totalQBlocks` 按各 batch 的 **存储** Q 长度（`cuSeqLengthsQ`）分块累加。
- 必须传入 `metadata`（由 `GenericBlockSparseAttentionMetadata` 生成，INT32 `[1024]`）以及 `blockTable`、`cuSeqLengthsQ`、`cuSeqLengthsKv`。
- `Nq` 必须能被 `Nkv` 整除（GQA）。
- Softmax 精度：
  - Ascend 950：仅支持 `softmaxPrecision=1`
  - Atlas A2/A3：`fp16` 支持 `0/1`；`bf16` 仅支持 `0`
- FP8 全量化：`quantType=5` 且 Q/K/V 为 `FLOAT8_E4M3FN`；此时不支持 `returnSoftmaxlse=1`。
- `sequsedQ` / `sequsedKv` 可选：任务空间按 actual 长度打包，GM/稀疏索引仍按 cu 存储偏移（段末 pad）。

### PAGED_BBND 非连续 KV（dim0）

- 允许 **仅首轴（物理页轴 dim0）** 非连续，页内 `blockSize × Nkv × D` 必须连续。
- aclnn **不会**对 `key`/`value` 做 `Contiguous` 拷贝；tiling 传入 `kStride0`/`vStride0`，kernel 按 `physicalId * stride0` 取页基址。
- `stride0` 须满足：`stride0 >= blockSize * Nkv * D`，且按 `Nkv * D` 对齐。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn API | 见 [aclnnGenericBlockSparseAttention](./docs/aclnnGenericBlockSparseAttention.md) | 两段式接口；需先跑 Metadata |
| PyTorch API | [test_torch_generic_block_sparse_attention.py](./examples/test_torch_generic_block_sparse_attention.py) | `npu_generic_block_sparse_attention_metadata` → `npu_generic_block_sparse_attention` |
| dim0 非连续冒烟 | [test_torch_gbsa_kv_dim0_strided.py](./examples/test_torch_gbsa_kv_dim0_strided.py) | 连续 KV vs dim0-strided KV 对照（需安装去掉 Contiguous(K/V) 的 opp） |

## 相关算子

- Metadata：[GenericBlockSparseAttentionMetadata](../sparse_attention_score_metadata/docs/aclnnGenericBlockSparseAttentionMetadata.md)
- 设计说明仓库内另有改造方案文档（开发参考，非对外接口规范）
