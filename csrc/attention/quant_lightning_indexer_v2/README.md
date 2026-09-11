# QuantLightningIndexerV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- API功能：QuantLightningIndexerV2是推理场景下，稀疏attention前处理的计算，选出关键的稀疏token，并对输入query和key进行量化实现存8算8，获取最大收益。

- 计算公式：
    $$out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\}$$
    主要计算过程为：
    1. 将某个token对应的输入参数`query`（$Q_{index}^{Quant}\in\R^{g\times d}$）乘以给定上下文`key`（$K_{index}^{Quant}\in\R^{S_{k}\times d}$），得到相关性。
    2. 相关性结果与`query`和`key`对应的反量化系数`query_dequant_scale`（$Scale_Q$）和`key_dequant_scale`（$Scale_K^T$）相乘，通过激活函数$ReLU$过滤无效负相关信号后，得到当前Token与所有前序Token的相关性分数向量。
    3. 将其与权重系数`weights`（$W$）相乘后，沿g的方向，选取前$Top-k$个索引值得到输出$out$，作为Attention的输入。

## 参数说明

| 参数名                     | 输入/输出/属性 | 描述  | 数据类型       | 数据格式   |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| query                     | 输入      | 公式中的$Q_{index}^{Quant}\in\R^{g\times d}$，表示输入Index Query，不支持非连续。| INT8、FLOAT8_e4m3fn、HIFLOAT8、FLOAT4_e2m1 | ND         |
| key                   | 输入      | 公式中的$K_{index}^{Quant}\in\R^{S_{k}\times d}$，表示压缩后的输入Index Key，支持0轴非连续。| INT8、FLOAT8_e4m3fn、HIFLOAT8、FLOAT4_e2m1 | ND |
| weights                 | 输入      | 公式中的$W$，表示权重系数，不支持非连续。 | FLOAT16、FLOAT32 | ND |
| query_dequant_scale             | 输入      | 公式中的$Scale_Q$，表示Index Query的反量化系数，不支持非连续。`quant_mode`为3/5时，shape为将`query`的D轴替换为(D/64, 2)；`quant_mode`为4时，shape为(1,)；其他场景shape与weights一致 | FLOAT16、FLOAT32、FLOAT8_e8m0     | ND         |
| key_dequant_scale            | 输入      | 公式中的$Scale_K$，表示Index Key的反量化系数，支持0轴非连续。`quant_mode`为3/5时，shape为将`key`的D轴替换为(D/64, 2)；`quant_mode`为4时，shape为(1,)；其他场景shape为移除`key`的D轴 | FLOAT16、FLOAT32、FLOAT8_e8m0       | ND         |
| cu_seqlens_q                    | 可选输入      | layout_q为TND时必须传入，表示每个Batch中`query`的有效token数前缀和。；layout_q为BSND时不能传入 | INT32       | ND         |
| cu_seqlens_k                    | 可选输入      | layout_k为TND时必须传入，表示每个Batch中`key`的有效token数前缀和；layout_k为PA_BSND或BSND时不能传入 | INT32       | ND         |
| seqused_q                    | 可选输入      | layout_q为BSND时可选传入，表示每个Batch中`query`的有效token数 | INT32       | ND         |
| seqused_k                    | 可选输入      | layout_k为PA_BSND或BSND时使用，表示每个Batch中`key`的有效token数。| INT32       | ND         |
| cmp_residual_k                    | 可选输入      | 压缩场景下Key的残余长度，需满足0 \<= cmp_residual_k\[i\] \< cmp_ratio。| INT32       | ND         |
| block_table                    | 可选输入      | 表示PageAttention中KV存储使用的block映射表。 | INT32       | ND         |
| output_idx_offset                    | 可选输入      | 输出索引的偏移量 | INT32       | ND         |
| metadata                    | 可选输入      | QuantLightningIndexerV2Metadata算子传入的分核信息，包含使用核数、分块大小以及每个核处理数据的起始点等内容。 | INT32       | ND         |
| quant_mode                 | 属性      | 用于标识输入的量化模式。 | INT32          | -         |
| max_seqlen_q                 | 可选属性| Query的最大序列长度，默认值-1表示任意可能长度 | INT32 | -         |
| layout_q                 | 可选属性| 用于标识输入`query`的数据排布格式，默认值"BSND"。 | STRING | -         |
| layout_k      | 可选属性      | 用于标识输入`key`的数据排布格式，默认值"BSND"。| STRING          | -         |
| topk  | 属性      | 代表topK阶段需要保留的索引数量，默认值2048。 | INT32          | -         |
| mask_mode | 可选属性      | 表示mask的模式，默认值0。 | INT32          | -         |
| cmp_ratio      | 可选属性      | 用于稀疏计算，表示key的压缩倍数，默认值1。 | INT32          | -         |
| return_value      |  可选属性     | 表示是否输出`sparse_values`，默认值0。 | INT32          | -         |
| sparse_indices     | 输出      | 公式中的输出Out，参与稀疏attention计算的token索引值。 | INT32          | ND         |
| sparse_values           | 输出      | 公式中的Indices输出对应的value值。`return_value`为1时shape与`sparse_indices`一致，`return_value`为0时shape为(0,) | BFLOAT16         | ND          |

## 约束说明

- <term>Ascend 950PR/Ascend 950DT</term>：
    - `query`、`key`在`quant_mode`为1/3时支持FLOAT8_e4m3fn，`quant_mode`为2时支持INT8，`quant_mode`为4时支持HIFLOAT8，`quant_mode`为5时支持FLOAT4_e2m1。
    - `query_dequant_scale`和`key_dequant_scale`在`quant_mode`为1/4时支持FLOAT32，`quant_mode`为2时支持FLOAT16，`quant_mode`为3/5时支持FLOAT8_e8m0。
    - `weights`在`quant_mode`为2时支持FLOAT16，`quant_mode`为1/3/4/5时支持FLOAT32。
    - `quant_mode`为3/5时，`query_dequant_scale`和`key_dequant_scale`的维数分别比`query`和`key`多1，前缀维度保持一致，末两维为(D/64,2)；D必须为64的倍数，每个scale对应连续32个D轴逻辑元素。
    - `query`的N支持[1, 64]，`key`的N仅支持1。
    - `topk`支持[1, 8192]。
    - 当传入的参数layout_query为TND时，必须传入cu_seqlens_q，如果也传入seqused_q，应保证由seqused_q传入的各个batch的query长度不超过根据cu_seqlens_q计算出的各个batch的q序列长度。当某个batch由seqused_q传入的q序列长度seqlen1小于由cu_seqlens_q计算出的query长度seqlen2时，会启用TND Padding功能，将该batch的seqlen2与seqlen1差值部分的query输出的sparse_indices和sparse_values全部置为无效值。部分长序列场景下，如果需要填充的无效数据过多，由于硬件限制可能会导致aicore执行超时，可以通过(seqlen2 - seqlen1) * topk来计算需要填充的数据量，建议将这个数据量控制在4亿以内。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - `quant_mode`仅支持2。
    - `query`、`key`支持INT8，不支持FLOAT8_e4m3fn、HIFLOAT8和FLOAT4_e2m1。
    - `query_dequant_scale`和`key_dequant_scale`支持FLOAT16，不支持FLOAT32和FLOAT8_e8m0。
    - `weights`支持FLOAT16，不支持FLOAT32。
    - 不支持`output_idx_offset`和`return_value`。
    - `query`的N仅支持64，`key`的N仅支持1。
    - `topk`支持[1, 2048]。

## 调用示例

| 调用方式 | 调用样例 | 说明 |
|----------|----------|------|
| PyTorch API | - | 通过[torch.ops.cann_ops_transformer.quant_lightning_indexer](../../torch_extension/cann_ops_transformer/docs/zh/quant_lightning_indexer.md)接口调用QuantLightningIndexerV2算子。 |
| aclnn API | [test_aclnn_quant_lightning_indexer_v2](examples/test_aclnn_quant_lightning_indexer_v2.cpp) | 通过[aclnnQuantLightningIndexerV2](docs/aclnnQuantLightningIndexerV2.md)两段式接口调用QuantLightningIndexerV2算子。 |
