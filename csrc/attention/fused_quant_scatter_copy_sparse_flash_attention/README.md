# FusedQuantScatterCopySparseFlashAttention

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- API功能：FusedQuantScatterCopySparseFlashAttention算子面向C8量化KV Cache场景，在一个MIX Kernel中完成DRAM到HBM的离散KV搬运及量化稀疏Flash Attention计算。
- 算子支持同一Batch内混合MTP0～15请求。每个请求通过`actual_seq_lengths_query`划分其Query区间，区间长度为1～16，分别对应MTP0～15。
- 当请求的`miss_counts`不大于512时，算子在Attention取数阶段融合处理miss数据；任一请求的`miss_counts`大于512时，算子在同一Kernel内先执行完整Scatter Copy，再执行Attention。
- Attention计算公式：

  $$
  O = \mathrm{softmax}(QK^T \cdot \alpha)V
  $$

  其中，$\alpha$对应接口参数`scale_value`，$K$和$V$来自C8量化KV Cache。每个KV token使用656字节封装，其中有效维度为512。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| query | 输入 | Query，shape为`[T, N, 576]`，其中`T`为Batch内所有请求的Query总数，`N`为Query Head数 | BFLOAT16、FLOAT16 | ND |
| hbm_kv_bytes | 输入/输出 | HBM侧C8量化KV Cache，shape为`[block_num_hbm, 128, 1, 656]`；Scatter Copy结果原地写入该Tensor | INT8 | ND |
| dram_kv_bytes | 输入 | DRAM侧C8量化KV Cache，shape为`[block_num_dram, 128, 1, 656]` | INT8 | ND |
| hbm_block_table | 输入 | HBM KV Cache的Page Attention块表，shape为`[B, max_block_num_hbm]` | INT32 | ND |
| dram_block_table | 输入 | DRAM KV Cache的Page Attention块表，shape为`[B, max_block_num_dram]` | INT32 | ND |
| topk_dst_slots | 输入 | 每个Query参与Attention的HBM token槽位，shape固定为`[T, 1, 2048]` | INT32 | ND |
| topk_src_ids | 输入 | 每个Query的miss项对应的DRAM token索引，shape固定为`[T, 1, 2048]` | INT32 | ND |
| topk_miss_counts | 输入 | 每个Query的有效miss数量，shape为`[T]`，取值范围为`[0, 2048]` | INT32 | ND |
| actual_seq_lengths_query | 输入 | 各请求Query结束位置的前缀和，shape为`[B]`，相邻值之差为1～16 | INT32 | ND |
| actual_seq_lengths_kv | 输入 | 各请求的实际KV序列长度，shape为`[B]` | INT32 | ND |
| num_cache_tokens | 输入 | 各请求卸载场景的HBM驻留token数量C，shape为`[B]`；非卸载场景取0 | INT32 | ND |
| miss_src_ids | 输入 | 请求级完整Scatter Copy的DRAM token索引，shape固定为`[B, 1, 32768]` | INT32 | ND |
| miss_dst_slots | 输入 | 请求级完整Scatter Copy的HBM token槽位，shape固定为`[B, 1, 32768]` | INT32 | ND |
| miss_counts | 输入 | 每个请求的有效Scatter Copy数量，shape为`[B]`，取值范围为`[0, 32768]` | INT32 | ND |
| scale_value | 属性 | Query与Key矩阵乘结果的缩放系数，必须为有限值 | FLOAT32 | - |
| attention_out | 输出 | Attention结果，shape为`[T, N, 512]`，数据类型与`query`一致 | BFLOAT16、FLOAT16 | ND |

## 约束说明

- 该接口支持推理场景和aclgraph模式。
- `query`的Head数仅支持8、16、32、64、128，Head维度固定为576；`attention_out`的Head维度固定为512。
- KV Cache仅支持C8规格，以INT8字节视图传入；block size固定为128，KV Head数固定为1，单token封装宽度固定为656字节。
- `hbm_block_table`和`dram_block_table`第二维取值范围为`[1, 2048]`，表内有效物理块编号不得越界。
- `topk_dst_slots`和`topk_src_ids`的最后一维固定为2048，有效数据位于每行前`topk_miss_counts`个位置。
- `miss_src_ids`和`miss_dst_slots`的最后一维固定为32768，有效数据位于每行前`miss_counts`个位置。
- Attention长度按请求选择：`num_cache_tokens`为0时使用`actual_seq_lengths_kv`，否则使用`num_cache_tokens`。因此同一Batch可混合卸载和非卸载请求。
- `num_cache_tokens`为0表示非卸载请求，此时`miss_counts`和对应Query的`topk_miss_counts`必须均为0；卸载请求的`num_cache_tokens`必须大于0。
- `actual_seq_lengths_query`必须严格递增，最后一个值必须等于`T`；每个请求的Query数量为1～16，对应MTP0～15。同一Batch允许不同请求使用不同MTP值。
- 当前接口不包含tail token参数，不单独建模尾块。
- 输入Tensor需位于同一设备。`query`、索引和长度Tensor必须连续；KV Cache允许使用Page Attention布局对应的非连续物理存储。

## vLLM Ascend接口

算子通过`_C_ascend`扩展注册，调用方负责预分配输出Tensor：

```python
attention_out = torch.empty(
    (query.shape[0], query.shape[1], 512),
    dtype=query.dtype,
    device=query.device,
)
torch.ops._C_ascend.npu_fused_quant_scatter_copy_sparse_flash_attention(
    query,
    hbm_kv_bytes,
    dram_kv_bytes,
    hbm_block_table,
    dram_block_table,
    topk_dst_slots,
    topk_src_ids,
    topk_miss_counts,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    num_cache_tokens,
    miss_src_ids,
    miss_dst_slots,
    miss_counts,
    scale_value,
    attention_out,
)
```

接口返回`None`。`hbm_kv_bytes`和`attention_out`分别以原地输入/输出Tensor的方式写入，支持
PyTorch Meta分发及aclgraph捕获。

## 编译与测试

在仓库`csrc`目录下可以仅编译该算子：

```bash
bash build.sh --pkg \
    --soc=ascend950 \
    --ops=fused_quant_scatter_copy_sparse_flash_attention
```

`csrc/build_aclnn.sh`的Ascend 950默认算子列表已经包含该算子。安装生成的自定义算子包和
vLLM Ascend扩展后，在仓库根目录运行最小功能测试：

```bash
python -m pytest -sv \
    tests/e2e/nightly/single_node/ops/singlecard_ops/\
test_fused_quant_scatter_copy_sparse_flash_attention.py
```

测试覆盖非卸载场景、`miss_counts <= 512`的融合miss路径和`miss_counts > 512`的算子内完整Scatter Copy
回退路径；同时覆盖`num_cache_tokens=32760`和目标槽位32759，并分别校验HBM Cache逐字节搬运结果和
Attention CPU golden精度。
