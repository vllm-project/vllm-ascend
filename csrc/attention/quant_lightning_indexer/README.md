# QuantLightningIndexer

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列加速卡产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- API功能：QuantLightningIndexer是推理场景下，稀疏attention前处理的计算，选出关键的稀疏token，并对输入query和key进行量化实现存8算8，获取最大收益。

- 计算公式：
    $$out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\}$$
    主要计算过程为：
    1. 将某个token对应的输入参数`query`（$Q_{index}^{Quant}\in\R^{g\times d}$）乘以给定上下文`key`（$K_{index}^{Quant}\in\R^{S_{k}\times d}$），得到相关性。
    2. 相关性结果与`query`和`key`对应的反量化系数`query_dequant_scale`（$Scale_Q$）和`key_dequant_scale`（$Scale_K^T$）相乘，通过激活函数$ReLU$过滤无效负相关信号后，得到当前Token与所有前序Token的相关性分数向量。
    3. 将其与权重系数`weights`（$W$）相乘后，沿g的方向，选取前$Top-k$个索引值得到输出$out$，作为Attention的输入。

## 参数说明

| 参数名                     | 输入/输出/属性 | 描述  | 数据类型       | 数据格式   |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| query                     | 输入      | 公式中的$Q_{index}\in\R^{g\times d}，表示输入Index Query$ | INT8、FLOAT8_e4m3fn | ND         |
| key                   | 输入      | 公式的$K_{index}\in\R^{S_{k}\times d}，表示压缩后的输入Index Key$ | INT8、FLOAT8_e4m3fn | ND |
| weights                 | 输入      | 公式中的$W$，表示权重系数，不支持非连续。| FLOAT16、FLOAT32 | ND |
| query_dequant_scale             | 输入      | 公式中的$Scale_Q$，表示Index Query的反量化系数，不支持非连续 | FLOAT16、FLOAT32     | ND         |
| key_dequant_scale            | 输入      | 公式中的$Scale_Q$，表示Index Key的反量化系数，不支持非连续 | FLOAT16、FLOAT32       | ND         |
| actual_seq_lengths_query                    | 可选输入      | 表示不同Batch中`query`的有效token数 | INT32       | ND         |
| actual_seq_lengths_key                    | 可选输入      | 表示不同Batch中`key`的有效token数 | INT32       | ND         |
| block_table                    | 可选输入      | 表示PageAttention中KV存储使用的block映射表 | INT32       | ND         |
| metadata                    | 可选输入      | QuantLightningIndexerMetadata算子传入的分核信息，包含使用核数、分块大小以及每个核处理数据的起始点等内容，shape大小为[1024]，当前不支持传空 | INT32       | ND         |
| query_quant_mode                 | 可选属性      | 用于标识输入`query`的量化模式，当前支持Per-Token-Head量化模式，当前仅支持传入0 | INT32          | -         |
| key_quant_mode                 | 可选属性| 用于标识输入`key`的量化模式，当前支持Per-Token-Head量化模式，当前仅支持传入0 | INT32 | -         |
| layout_query                 | 可选属性| 用于标识输入`query`的数据排布格式，当前支持BSND、TND，默认值"BSND" | STRING | -         |
| layout_key      | 可选属性      | 用于标识输入`key`的数据排布格式，当前仅支持传入PA_BSND  | STRING          | -         |
| sparse_count  | 可选属性      | 代表topK阶段需要保留的block数量，Atlas A3 推理系列产品支持[1, 2048]，Ascend 950PR/Ascend 950DT支持512 | INT32          | -         |
| sparse_mode | 可选属性      | 表示sparse的模式，支持0/3，数据类型支持`int32`。 sparse_mode为0时，代表defaultMask模式。sparse_mode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。 | INT32          | -         |
| pre_tokens    | 可选属性      | 预留参数，表示attention需要和前几个Token计算关联，仅支持默认值2^63-1 | INT64          | -         |
| next_tokens    | 可选属性      | 预留参数，表示attention需要和前几个Token计算关联，仅支持默认值2^63-1 | INT64          | -         |
| cmp_ratio      | 可选属性      | 用于稀疏计算，表示key的压缩倍数。数据类型支持`int32`。Atlas A3 推理系列产品支持1/2/4/8/16/32/64/128，Ascend 950PR/Ascend 950DT支持1/4/128。 | INT32          | -         |
| return_value      |  可选属性     | 表示是否输出`sparse_values`。True表示输出，False表示不输出；仅支持默认值False | BOOL          | -         |
| stride      |  可选属性     | 表示key的首轴的stride | INT32          | -         |
| sparse_indices     | 输出      | 公式中的输出Out，参与稀疏attention计算的token索引值 | INT32          | ND         |
| sparse_values           | 输出      | 公式中的Indices输出对应的value值，**目前暂不支持返回sparse_values。** | FLOAT32         | ND          |

- <term>Ascend 950PR/Ascend 950DT</term>：query、key不支持INT8；weights、query_dequant_scale和key_dequant_scale不支持FLOAT16。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：query、key不支持FLOAT8_e4m3fn；weights、query_dequant_scale和key_dequant_scale不支持FLOAT32。

## 约束说明

- 该接口支持图模式。
- 该接口要求$W \odot Scale_Q$的结果在`float16`(Atlas A3)/`float32`(Ascend 950PR/Ascend 950DT)的表示范围内。
- 该接口的TopK过程对NAN排序是未定义行为。
- 参数query中的D轴和参数key中的D轴值相等为128。
- 参数query和key中的N轴分别仅支持64和1。
- 当`layout_query`为TND时，`actual_seq_lengths_query`必须传入，且以该入参元素的数量作为B值，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。不能出现负值。
- 当`layout_key`为PA_BSND时，`actual_seq_lengths_key`该入参必须传入。
- PageAttention场景下，`block_table`必须为二维，第一维长度需要等于B，第二维长度不能小于maxBlockNumPerSeq(maxBlockNumPerSeq为每个batch中最大`actual_seq_lengths_key`对应的block数量)，支持block_size取值为16的整数倍，最大支持到1024。
- query、key、weights、query_dequant_scale、key_dequant_scale数据排布格式支持从多种维度解读，其中B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。

## Atlas A3 推理系列产品 调用说明

- 单算子模式调用
    ```python
    import torch
    import torch_npu
    import numpy as np
    import torch.nn as nn
    import math
    import custom_ops

    n1 = 64
    n2 = 1
    d = 128
    block_size = 128
    layout_key = "PA_BSND"
    layout_query = "BSND"
    query_quant_mode = 0
    key_quant_mode = 0
    np.random.seed(0)
    # -------------
    b = 24
    t = None
    s1 = 4
    s2 = 512
    act_seq_q = None
    act_seq_k = None
    sparse_mode = 0
    sparse_count = 512
    cmp_ratio = 1
    max_block_table_num = (s2 + block_size - 1) // block_size
    block_table = torch.tensor([range(b * max_block_table_num)], dtype = torch.int32).reshape(b, -1)
    key = torch.tensor(np.random.uniform(-128, 127, (b * max_block_table_num, block_size, n2, d))).to(torch.int8)
    key_dequant_scale = torch.tensor(np.random.uniform(0, 10, (b * max_block_table_num, block_size, n2)))
    key_dequant_scale = key_dequant_scale.to(torch.float16)
    query = torch.tensor(np.random.uniform(-128, 127, (b, s1, n1, d))).to(torch.int8)
    query_dequant_scale = torch.tensor(np.random.uniform(0, 10, (b, s1, n1))).to(torch.float16)
    weights = torch.tensor(np.random.uniform(0, 0.01, (b, s1, n1))).to(torch.float16)
    actual_seq_lengths_query = torch.tensor(np.random.uniform(s1, s1, (b))).to(torch.int32) \
                                if act_seq_q is None else torch.tensor(act_seq_q).to(torch.int32)
    actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(torch.int32) \
                                if act_seq_k is None else torch.tensor(act_seq_k).to(torch.int32)
    max_seqlen_q = actual_seq_lengths_query.max().item()
    max_seqlen_k = actual_seq_lengths_key.max().item()
    metadata = torch.ops.custom.npu_quant_lightning_indexer_metadata (
                                    actual_seq_lengths_query = actual_seq_lengths_query.npu(),
                                    actual_seq_lengths_key = actual_seq_lengths_key.npu(),
                                    num_heads_q = n1,
                                    num_heads_k = n2,
                                    head_dim = d,
                                    query_quant_mode = query_quant_mode,
                                    key_quant_mode = key_quant_mode,
                                    batch_size = b,
                                    max_seqlen_q = max_seqlen_q,
                                    max_seqlen_k = max_seqlen_k,
                                    layout_query = layout_query,
                                    layout_key = layout_key,
                                    sparse_count = sparse_count,
                                    sparse_mode = sparse_mode,
                                    pre_tokens = (1<<63)-1,
                                    next_tokens = (1<<63)-1,
                                    cmp_ratio = cmp_ratio,
                                    device = 'npu:0')

    sparse_indices, sparse_values = torch.ops.custom.npu_quant_lightning_indexer(query.npu(), key.npu(), weights.npu(), query_dequant_scale.npu(),
                                                    key_dequant_scale.npu(),
                                                    actual_seq_lengths_query=actual_seq_lengths_query.npu(),
                                                    actual_seq_lengths_key=actual_seq_lengths_key.npu(),
                                                    block_table=block_table.npu(),
                                                    metadata = metadata,
                                                    query_quant_mode=query_quant_mode,
                                                    key_quant_mode=key_quant_mode,
                                                    layout_query=layout_query,
                                                    layout_key=layout_key, sparse_count=sparse_count,
                                                    sparse_mode=sparse_mode, pre_tokens=(1<<63)-1,
                                                    next_tokens=(1<<63)-1, cmp_ratio=cmp_ratio)
    ```
- aclgarph调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    import torch.nn as nn
    import math
    import torchair
    import custom_ops
    from torchair.configs.compiler_config import CompilerConfig

    n1 = 64
    n2 = 1
    d = 128
    block_size = 128
    layout_key = "PA_BSND"
    layout_query = "BSND"
    query_quant_mode = 0
    key_quant_mode = 0
    np.random.seed(0)
    # -------------
    b = 24
    t = None
    s1 = 4
    s2 = 512
    act_seq_q = None
    act_seq_k = None
    sparse_mode = 3
    sparse_count = 512
    pre_tokens=(1<<63)-1
    next_tokens=(1<<63)-1
    cmp_ratio = 4
    max_block_table_num = (s2 + block_size - 1) // block_size
    block_table = torch.tensor([range(b * max_block_table_num)], dtype = torch.int32).reshape(b, -1).npu()
    key = torch.tensor(np.random.uniform(-128, 127, (b * max_block_table_num, block_size, n2, d))).to(torch.int8).npu()
    key_dequant_scale = torch.tensor(np.random.uniform(0, 10, (b * max_block_table_num, block_size, n2))).npu()
    key_dequant_scale = key_dequant_scale.to(torch.float16).npu()
    query = torch.tensor(np.random.uniform(-128, 127, (b, s1, n1, d))).to(torch.int8).npu()
    query_dequant_scale = torch.tensor(np.random.uniform(0, 10, (b, s1, n1))).to(torch.float16).npu()
    weights = torch.tensor(np.random.uniform(0, 0.01, (b, s1, n1))).to(torch.float16).npu()
    actual_seq_lengths_query = torch.tensor(np.random.uniform(s1, s1, (b))).to(torch.int32).npu() \
                                if act_seq_q is None else torch.tensor(act_seq_q).to(torch.int32).npu()
    actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(torch.int32).npu() \
                                if act_seq_k is None else torch.tensor(act_seq_k).to(torch.int32).npu()
    max_seqlen_q = actual_seq_lengths_query.max().item()
    max_seqlen_k = actual_seq_lengths_key.max().item()

    class QLINetwork(nn.Module):
        def __init__(self):
            super(QLINetwork, self).__init__()

        def forward(self, query, key, weights, q_scale, k_scale, query_quant_mode, key_quant_mode,
                    batch_size, num_heads_q, num_heads_k, head_dim,
                    actual_seq_lengths_query=None, actual_seq_lengths_key=None,
                    block_table=None, layout_query='BSND', layout_key='BSND',
                    sparse_count=512, sparse_mode=3, pre_tokens=(1<<63)-1,
                    next_tokens=(1<<63)-1, cmp_ratio=cmp_ratio, return_value=False):
            metadata = torch.ops.custom.npu_quant_lightning_indexer_metadata(
                                    actual_seq_lengths_query = actual_seq_lengths_query,
                                    actual_seq_lengths_key = actual_seq_lengths_key,
                                    num_heads_q = num_heads_q,
                                    num_heads_k = num_heads_k,
                                    head_dim = head_dim,
                                    query_quant_mode = query_quant_mode,
                                    key_quant_mode = key_quant_mode,
                                    batch_size = batch_size,
                                    max_seqlen_q = max_seqlen_q,
                                    max_seqlen_k = max_seqlen_k,
                                    layout_query = layout_query,
                                    layout_key = layout_key,
                                    sparse_count = sparse_count,
                                    sparse_mode = sparse_mode,
                                    pre_tokens = (1<<63)-1,
                                    next_tokens = (1<<63)-1,
                                    cmp_ratio = cmp_ratio,
                                    device = 'npu:0')

            sparse_indices, sparse_values = torch.ops.custom.npu_quant_lightning_indexer(query, key, weights,
                                                        q_scale, k_scale,
                                                        actual_seq_lengths_query=actual_seq_lengths_query,
                                                        actual_seq_lengths_key=actual_seq_lengths_key,
                                                        block_table=block_table, metadata=metadata,
                                                        query_quant_mode=query_quant_mode,
                                                        key_quant_mode=key_quant_mode,
                                                        layout_query=layout_query,
                                                        layout_key=layout_key, sparse_count=sparse_count,
                                                        sparse_mode=sparse_mode,pre_tokens=pre_tokens,
                                                        next_tokens=next_tokens, cmp_ratio=cmp_ratio,
                                                        return_value=return_value)
            return sparse_indices


    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    npu_mode = torch.compile(QLINetwork().npu(), fullgraph=True, backend=npu_backend, dynamic=False)
    sparse_indices = npu_mode( query, key, weights, query_dequant_scale, key_dequant_scale,
                        query_quant_mode, key_quant_mode, b, n1, n2, d,
                        actual_seq_lengths_query=actual_seq_lengths_query,
                        actual_seq_lengths_key=actual_seq_lengths_key,
                        block_table=block_table,
                        layout_query=layout_query, layout_key=layout_key,
                        sparse_count=sparse_count, sparse_mode=sparse_mode,
                        pre_tokens=pre_tokens, next_tokens=next_tokens,
                        cmp_ratio=cmp_ratio, return_value=False)
    ```

更多使用示例见[pytest示例](./tests/pytest/README.md)。
