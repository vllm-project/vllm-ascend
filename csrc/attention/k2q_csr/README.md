# k2q_csr

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>                        | √  |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>        | √  |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>        | √  |
|<term>Atlas 200I/500 A2 推理系列产品</term>                    | ×  |
|<term>Atlas 推理系列产品</term>                                | ×  |
|<term>Atlas 训练系列产品</term>                                | ×  |

## 功能说明

- API功能：`k2q_csr`算子将稀疏注意力中的 Q→K 映射索引（`q2k`）转换为 K→Q 的 CSR（Compressed Sparse Row）索引结构，供后续稀疏注意力 score 算子（如`npu_sparse_attention_score_prefill`）使用。该算子是 MiniMax-M3 稀疏注意力在 A3/A2 平台上的索引预处理算子。

- 计算公式：

    输入`q2k` ∈ ℤ^{H×T×K}，其中`q2k[h][t][k]` ∈ {-1} ∪ {0, …, R-1} 表示头 h 下查询 token t 的第 k 个选中项所映射到的 KV block 行号，-1 表示无效/填充。

    输出 CSR 三元组：

    $$
    \text{row\_ptr}[h][r+1] = \text{row\_ptr}[h][r] + \sum_{t=0}^{T-1}\sum_{k=0}^{K-1}\mathbb{1}[\,\text{q2k}[h][t][k] = r\,]
    $$

    其中`row_ptr`为 CSR 行指针（前缀和），`q_ind[h][i]`为第 i 个非零项对应的查询 token 索引，`slot[h][i]`为该项在`q2k` topk 维度中的位置（0 ≤ slot < K）。`q_ind`与`slot`的无效项以 -1 填充。

- 计算阶段：算子内部由五阶段 AscendC 算子串联完成，依次为 Meta → Hist → RowPrefix → TilePrefix → Scatter。

## 参数说明

| 参数名            | 输入/输出/属性 | 描述                                                                                                          | 数据类型 | 数据格式 |
|-------------------|-----------|-------------------------------------------------------------------------------------------------------------|----------|----------|
| q2k               | 输入      | Q→K 映射索引，对应公式中的`q2k`，shape 为[H, T, topk]，-1 表示无效填充。                                            | INT32    | ND       |
| cu\_seqlens       | 输入      | 各 Batch 的 Q token 累计前缀和，shape 为[B+1]，后一元素 ≥ 前一元素。                                                | INT32    | ND       |
| cu\_block\_lens   | 输入      | 各 Batch 的逻辑 KV block 累计前缀和，shape 为[B+1]，其末元素即`total_rows`。                                         | INT32    | ND       |
| order\_method     | 可选属性  | 行打包顺序，0=batch/concat（按 batch 拼接），1=round-robin（跨 batch 轮询），默认值为0。                                  | INT32    | -        |
| total\_rows       | 可选属性  | KV block 总行数（=`cu_block_lens[-1]`）。≥0 时直接使用；<0 时由 Host 端 D2H 推导（有同步开销，推荐显式传入）。              | INT32    | -        |
| max\_kv           | 可选属性  | 单 Batch 最大 KV block 数。≥0 时直接使用；<0 时由 Host 端 D2H 推导（有同步开销，推荐显式传入）。                          | INT32    | -        |
| use\_simt         | 可选属性  | ascend950 上 Hist/Scatter 阶段是否走 SIMT 路径。0=`K2qCsrPipelineMc`（SIMD/MC，A2+A5），1=SIMT VF（仅 ascend950），默认值为0。 | INT32    | -        |
| q\_global\_offset | 可选属性  | `q_ind` 的索引语义。0=batch-local（`q_ind = qAbs - cu_q[bi]`），1=全局 Q 下标（`q_ind = qAbs`），默认值为0。              | INT32    | -        |
| row\_ptr          | 输出      | CSR 行指针，对应公式中的`row_ptr`，shape 为[H, total_rows+1]。                                                       | INT32    | ND       |
| q\_ind            | 输出      | 查询 token 索引，对应公式中的`q_ind`，shape 为[H, T*topk]，无效项以 -1 填充。                                          | INT32    | ND       |
| slot              | 输出      | topk 内位置索引，对应公式中的`slot`，shape 为[H, T*topk]，取值范围[0, topk)，无效项以 -1 填充。                           | INT32    | ND       |

## 约束说明

- 该接口支持推理（prefill）场景下使用。
- 该接口支持aclgraph模式。
- `q2k`必须为 3-D [H, T, topk]，且数据类型为 INT32。
- `cu_seqlens`必须为 1-D [B+1]，且为递增前缀和（首元素为 0）。
- `cu_block_lens`必须为 1-D [B+1]，且为递增前缀和（首元素为 0），其末元素即`total_rows`。
- `order_method`仅支持输入 0 或 1。
- `total_rows`与`max_kv`必须 ≥ 0；当传入 <0 时，Host 端会对`cu_block_lens`执行 D2H 同步推导，产生额外同步开销，推荐显式传入。
- `use_simt`仅在 ascend950（A3）上生效；在非 950 平台（如 ascend910b）上由 tiling 强制置 0，走 MC（SIMD）路径。
- `q_global_offset`仅影响 Scatter 阶段的`q_ind`语义，其余阶段固定为 0。
- 输出`slot`的最大值一定 < topk。
- `row_ptr`初始化为 0，`q_ind`与`slot`初始化为 -1。
- 维度说明：H（Head-Num）表示头数，T 表示所有 Batch 的 Q token 总数，topk 表示每个 Q token 选取的 KV block 数，B（Batch）表示输入样本批量大小，R（=total_rows）表示 KV block 总行数。
- 源码对齐`xpu_kernel/C_like/transformer/npu/kvcache/k2q_csr`（含 MC FastB1 / 批量 MTE3 / qTile ping-pong、`q_global_offset`、SIMT 路径隔离）。

## 调用说明

- 单算子模式调用

    ```python
    import torch
    import vllm_ascend.vllm_ascend_C  # noqa: F401  注册自定义算子

    H, T, topk = 2, 13, 4
    B = 2
    # q2k[h, t, k] ∈ {-1} ∪ {0, …, R-1}，-1 为无效填充
    q2k = torch.randint(-1, 18, (H, T, topk), dtype=torch.int32).npu()
    # 各 batch 的 Q token 前缀和与 KV block 前缀和
    cu_seqlens = torch.tensor([0, 5, 13], dtype=torch.int32).npu()
    cu_block_lens = torch.tensor([0, 9, 18], dtype=torch.int32).npu()

    total_rows = int(cu_block_lens[-1].item())
    block_lens = cu_block_lens[1:] - cu_block_lens[:-1]
    max_kv = int(block_lens.max().item())

    row_ptr, q_ind, slot = torch.ops._C_ascend.npu_k2q_csr(
        q2k,
        cu_seqlens,
        cu_block_lens,
        order_method=1,        # 0=batch/concat, 1=round-robin
        total_rows=total_rows, # ≥0 推荐显式传入，避免 Host D2H
        max_kv=max_kv,         # ≥0 推荐显式传入
        use_simt=0,            # ascend950 可置 1 走 SIMT；0=MC(A2/A5)
        q_global_offset=True,  # False=batch-local q_ind；True=全局 Q 下标
    )
    ```

    底层等价于上述`torch.ops._C_ascend.npu_k2q_csr(...)`，亦可使用 vllm-ascend 内置封装：

    ```python
    from vllm_ascend.models.minimax_m3.ops.msa_m3_npu import _npu_k2q_csr

    row_ptr, q_ind, slot = _npu_k2q_csr(
        q2k,
        cu_seqlens,
        cu_block_lens,
        order_method=1,
        total_rows=total_rows,
        max_kv=max_kv,
        use_simt=0,
        q_global_offset=True,
    )
    ```

    Host 编排实现见`k2q_csr_torch_adpt.h`，对应五阶段算子：`aclnnK2qCsrMeta` → `aclnnK2qCsrHist` → `aclnnK2qCsrRowPrefix` → `aclnnK2qCsrTilePrefix` → `aclnnK2qCsrScatter`。

## 目录

- `k2q_csr_meta/`：Meta 阶段，生成 row_map 与 token_batch workspace。
- `k2q_csr_hist/`：Hist 阶段，统计每个 KV 行的命中计数。
- `k2q_csr_row_prefix/`：RowPrefix 阶段，行内前缀和并写`row_ptr`。
- `k2q_csr_tile_prefix/`：TilePrefix 阶段，tile 级前缀和。
- `k2q_csr_scatter/`：Scatter 阶段，写`q_ind`与`slot`。
- `k2q_csr_common/`：五阶段共享 tiling / kernel 源（构建时自动 vendor 至各阶段`op_kernel/common/`，该目录 gitignore，勿手工依赖）。
- `k2q_csr_torch_adpt.h`：`torch.ops._C_ascend.npu_k2q_csr` 的 Host 编排。

## 打包编译（含 kernel binary）

```bash
source /usr/local/Ascend/cann/set_env.sh   # 或本机 cann/ascend-toolkit 路径
cd /path/to/vllm-ascend/csrc

# A3（ascend950）；A2 用 --soc=ascend910b
bash build.sh --pkg --soc=ascend950 --ops=k2q_csr -j$(nproc)
bash build_out/cann-*-custom_linux-*.run --quiet
```

CMake 配置时会自动将`k2q_csr_common/op_kernel`同步到各阶段`op_kernel/common/`（该目录 gitignore，勿手工依赖已 vendor 的副本）。

精度测试：

```bash
pytest tests/ut/ops/test_k2q_csr.py -v
```
