# MsaIndexScore

## 产品支持情况

| 产品                                                      | 是否支持 |
| --------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品</term> |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  |    √     |
| <term>Ascend 950PR/Ascend 950DT</term>                    |    ×     |

## 功能说明

- **算子功能**：计算 MSA（MiniMax Sparse Attention）模块 Index Branch 中的 block score。对每个 query token 与每个 KV sparse block，取该 block 内所有因果可见 token 的 $Q_{idx}$ 和 $K_{idx}$（可选 int8 反量化）的"matmul+maxpool"运算，得到逐 block 的重要性分数 `score`，用作 Index Branch 中后续 TopK 的输入。Prefill 与 Decode 由同一接口承载。

- **计算公式**：

    - 非量化场景：

    $$
    score = Maxpool[ Q_{idx}@K_{idx}^{T} ]
    $$

    - int8 量化场景：

    $$
    score = Maxpool[ scale \cdot Q_{idx}@K_{idx}^{T} ]
    $$

    完整公式：

    $$
    score = Maxpool[(scale \cdot) Q_{idx}@K_{idx}^{T} + atten\_mask] + local\_mask
    $$

    其中 Maxpool 按 sparse block（长度为 $block\_size$）在 KV token 维上取最大值。`start_loc`、`init_blocks`、`local_blocks` 共同生成 $local\_mask$，对序列头部 / 当前 query 附近若干强制保留的 block 写入高分，保证后续 TopK 一定选中它们。与 Triton raw score kernel 对齐时可将 `init_blocks`、`local_blocks` 置 0，关闭 $local\_mask$。

## 参数说明

> **说明：**
>
> - B（Batch Size）表示输入样本批量大小
> - S（Sequence Length）表示序列长度，$S1$ 为 query 侧、$S2$ 为 key 侧
> - T 表示所有 Batch 序列长度累加和，$T1$ 为 query 侧、$T2$ 为 key 侧
> - N（Head Num）表示头数，$N1$ 为 query 侧、$N2$ 为 key 侧
> - D（Head Dim）表示单个注意力头维度
> - PageAttention 场景下 $block\_num$ 为物理 block 总数、$block\_size$ 为每个 block 的 token 数，$maxBlockNumPerSeq$ 为每个 batch 最大逻辑 block 数（通常 $\ge\lceil S2/block\_size\rceil$），$M_b=\lceil S2/block\_size\rceil$ 为逻辑 block 总数

| 参数名                   | 输入/输出/属性 | 描述                                                         | 数据类型                     | 数据格式 |
| ------------------------ | -------------- | ------------------------------------------------------------ | ---------------------------- | -------- |
| query                    | 输入           | 公式中的 $Q_{idx}$。当前仅支持 TND，shape 为 $[T1, N1, D]$   | BFLOAT16, FLOAT16            | ND       |
| key                      | 输入           | 公式中的 $K_{idx}$。支持 TND（$[T2, N2, D]$）、BNBD（$[block\_num, N2, block\_size, D]$）、BBND（$[block\_num, block\_size, N2, D]$） | BFLOAT16, FLOAT16, INT8      | ND, NZ   |
| block_table              | 可选输入       | PageAttention 的逻辑 block → 物理 page 映射表。PA 场景必须传入，二维，第二维长度不能小于 $maxBlockNumPerSeq$；shape 为 $[B, S2/block\_size]$ | INT32                        | ND       |
| scale                    | 可选输入       | 公式中的 $scale$，反量化系数。非量化必须为空；量化场景必选。PA 为 $[block\_num, N2, block\_size]$ 或 $[block\_num, block\_size, N2]$；TND 为 $[T2, N2]$ | FLOAT                        | ND, NZ   |
| atten_mask               | 可选输入       | 控制因果可见的 mask。仅在 `sparse_mode=3` 时使用；取值为 1 表示该位不参与计算，为 0 表示参与计算；shape 为 $[2048, 2048]$ | INT8                         | ND       |
| actual_seq_qlen | 可选输入       | 每个 Batch 中 Query 的有效 token 数。query 为 TND 时必须传入，单调不减（前缀和），shape 为 $[B+1]$ | INT32                        | ND       |
| actual_seq_klen   | 可选输入       | 每个 Batch 中 Key 的有效 token 数。key 为 TND 时必须传入（前缀和）；PageAttention 场景下为各请求可见 $S2$，shape 为 $[B]$ | INT32                        | ND       |
| start_loc                | 输入           | 当前 query 所在逻辑 block 索引（非 token 前缀），用于生成 $local\_mask$；shape 为 $[B]$ | INT32                        | ND       |
| layout_key               | 属性           | key 布局。`"TND"` / `"BBND"` / `"BNBD"`。aclnn 参数名为 `layoutKeyOptional`，不传时默认 `"BBND"` | STRING                       | -        |
| sparse_mode              | 属性           | sparse 模式。0：defaultMask（`atten_mask` 传空）；3：rightDownCausal（须传入 $[2048, 2048]$ 的 `atten_mask`） | INT64                        | -        |
| init_blocks              | 属性           | $local\_mask$ 强制选中的头部 block 数。对逻辑 block $[0, init\_blocks)$ 写入高分 $1\mathrm{e}30$。可选，默认 $0$ | INT64                        | -        |
| local_blocks             | 属性           | $local\_mask$ 强制选中的局部窗口长度。窗口为 $[max(0, start\_loc+1-local\_blocks), start\_loc]$，写入高分 $1\mathrm{e}29$（覆盖同位置的 `init_blocks`）。可选，默认 $1$（对齐 MiniMax HF）；与 Triton raw score 对齐时置 $0$ | INT64                        | -        |
| score                    | 输出           | 公式中的 $score$，逐 block 重要性分数；shape 为 $[N1, T1, RoundUp(maxBlockNumPerSeq, 16)]$ | FLOAT                        | ND       |

## 约束说明

- 当前 $block\_size$ 仅支持 128。
- `layout_key` 必须显式指定：`"BBND"` / `"BNBD"` / `"TND"`，与 `key` 实际 shape 一致。
- PageAttention（`layout_key` 为 `"BBND"` / `"BNBD"`）场景下，`block_table` 必须传入；TND key 场景不得传入 `block_table`，`actual_seq_klen` 为 `[B+1]` 前缀和。
- 非量化场景下，`key` dtype 与 `query` 相同（当前为 BFLOAT16 / FLOAT16），`scale` 必须为空；量化场景下仅支持 INT8，`scale` 必选：PA 为 $[block\_num, N2, block\_size]$ 或 $[block\_num, block\_size, N2]$，TND 为 $[T2, N2]$，dtype 为 FLOAT。当前不支持 FP8 与 <term>Ascend 950PR/Ascend 950DT</term>。
- `sparse_mode` 当前仅支持 0、3：
    - 为 0 时，代表 defaultMask 模式，`atten_mask` 传入空；
    - 为 3 时，代表 rightDownCausal 模式，`atten_mask` 必须传入，shape 为 $[2048, 2048]$，取值为 1 代表该位不参与计算，为 0 代表该位参与计算。
- `init_blocks`、`local_blocks` 必须 $\ge 0$ 且不超过逻辑 block 数（PA 为 `block_table` 第二维；TND 为 score 末维对齐宽度）。两者均为 0 时跳过 $local\_mask$。
- 本算子输出止于 block score，**不包含** TopK。

## 调用示例

| 调用方式 | 样例代码 | 说明 |
|----------|----------|------|
| aclnn 单算子调用 | [test_aclnn_msa_index_score.cpp](./examples/test_aclnn_msa_index_score.cpp) | 内置 CPU golden 的端到端精度自验证 |
| 接口文档 | [aclnnMsaIndexScore.md](./docs/aclnnMsaIndexScore.md) | 两段式接口说明 |
| 测试说明 | [tests/README.md](./tests/README.md) | 用例矩阵与运行方式 |

编译与运行：

```bash
bash build.sh --pkg --soc=ascend910b --ops=msa_index_score -j32
./build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/tmp/msa_opp
export ASCEND_CUSTOM_OPP_PATH=/tmp/msa_opp/vendors/custom_transformer
bash build.sh --run_example msa_index_score eager cust --vendor_name=custom
```

> **实现备注（A2/A3）**
>
> - key 布局由属性 `layout_key`（aclnn：`layoutKeyOptional`）指定，支持 PageAttention **BBND** / **BNBD**，以及 packed **TND**（无 `block_table`，`actual_seq_klen` 为 `[B+1]` 前缀和）。默认 `"BBND"`。
> - `sparse_mode=3` 的 `atten_mask[2048,2048]` 在 host 校验必选；device 侧按 rightDownCausal
>   解析可见窗口（与 LightningIndexer 一致），不逐元素加载模板。
> - `start_loc` 为逻辑 block 索引，与属性 `init_blocks`（默认 0）、`local_blocks`（默认 1）
>   一起在 Maxpool 之后施加 `local_mask`。
> - 完整公式：`score = Maxpool[(scale·)Q@Kᵀ + atten_mask] + local_mask`。
