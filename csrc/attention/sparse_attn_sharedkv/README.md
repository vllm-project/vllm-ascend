# SparseAttnSharedkv

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>                        | ×  |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>        | √  |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>        | √  |
|<term>Atlas 200I/500 A2 推理系列产品</term>                    | ×  |
|<term>Atlas 推理系列产品</term>                                | ×  |
|<term>Atlas 训练系列产品</term>                                | ×  |

## 功能说明

- API功能：`SparseAttnSharedKV`算子旨在完成以下公式描述的Attention计算，支持Sliding Window Attention、Compressed Attention以及Sparse Compressed Attention。

- 计算公式：

    $$
    O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
    $$

    其中$\tilde{K}=\tilde{V}$为基于ori_kv、cmp_kv以及cmp_ratio等入参控制的实际参与计算的 $KV$。

## 参数说明

| 参数名            | 输入/输出/属性 | 描述  | 数据类型       | 数据格式   |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| q                     | 输入      | 对应公式中的$Q$。                                                     | BFLOAT16、FLOAT16  | ND |
| ori\_kv               | 可选输入  | 对应公式中的$\tilde{K}和\tilde{V}$的一部分，为原始不经压缩的KV。          | BFLOAT16、FLOAT16 | ND |
| cmp\_kv               | 可选输入  | 对应公式中的$\tilde{K}和\tilde{V}$的一部分，为经过压缩的KV。             | BFLOAT16、FLOAT16  | ND |
| ori\_sparse\_indices  | 可选输入  | 代表离散取oriKvCache的索引。                                           | INT32              | ND |
| cmp\_sparse\_indices  | 可选输入  | 代表离散取cmpKvCache的索引。                                           | INT32              | ND |
| ori\_block\_table     | 可选输入  | 表示PageAttention中oriKvCache存储使用的block映射表。                   | INT32               | ND |
| cmp\_block\_table     | 可选输入  | 表示PageAttention中cmpKvCache存储使用的block映射表。                   | INT32               | ND |
| cu\_seqlens\_q        | 可选输入  | 表示不同Batch中`q`的有效token数。                                      | INT32               | ND |
| cu\_seqlens\_ori\_kv  | 可选输入  | 表示不同Batch中`ori_kv`的有效token数。                                 | INT32               | ND |
| cu\_seqlens\_cmp\_kv  | 可选输入  | 表示不同Batch中`cmp_kv`的有效token数。                                 | INT32               | ND |
| seqused\_q            | 可选输入  | 表示不同Batch中`q`实际参与运算的token数。                               | INT32               | ND |
| seqused\_kv           | 可选输入  | 表示不同Batch中`ori_kv`实际参与运算的token数。                          | INT32               | ND |
| sinks                 | 可选输入  | 注意力下沉tensor。                                                     | FLOAT32             | ND |
| metadata              | 可选输入  | aicpu算子（npu\_sparse\_attn\_sharedkv\_metadata）的分核结果。          | INT32               | ND |
| softmax\_scale        | 可选属性  | 代表缩放系数，对应公式中的$\text{softmax\_scale}$，默认值为None。         | FLOAT32             | - |
| cmp_ratio             | 可选属性  | 表示对`ori_kv`的压缩率，仅支持输入4或128，默认值为None。                 | INT32               | - |
| ori\_mask\_mode       | 可选属性  | 表示`q`和`ori_kv`计算的mask模式，仅支持输入默认值4。                     | INT32               | - |
| cmp\_mask\_mode       | 可选属性  | 表示`q`和`cmp_kv`计算的mask模式，仅支持输入默认值3。                     | INT32               | - |
| ori\_win\_left        | 可选属性  | 表示`q`和`ori_kv`计算中q对过去token计算的数量，仅支持输入默认值127。      | INT32               | - |
| ori\_win\_right       | 可选属性  | 表示`q`和`ori_kv`计算中q对未来token计算的数量，仅支持输入默认值0。        | INT32               | - |
| layout\_q             | 可选属性  | 用于标识输入`q`的数据排布格式，支持输入"TND"和"BSND"，默认值为"BSND"。     | STRING               | - |
| layout\_kv            | 可选属性  | 用于标识输入`ori_kv`和`cmp_kv`的数据排布格式，支持输入"PA_ND"和"BSND"。   | STRING               | - |
| return\_softmax_lse   | 可选属性  | 表示是否返回`softmax_lse`。True表示返回，False表示不返回，默认值为False。 | BOOL                | -  |
| attention\_out        | 输出      | 公式中的输出。                                                          | BFLOAT16、FLOAT16   | ND |
| softmax\_lse          | 输出      | 返回的`softmax_lse`。                                                 | FLOAT32             | ND |

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持aclgraph模式。
- 该接口当前支持三种计算场景：场景一，仅传入`ori_kv`时为Sliding Window Attention计算；场景二，传入`ori_kv`及`cmp_kv`时为Sliding Window Attention + Compressed Attention计算；场景三，传入`ori_kv`、`cmp_kv`及`cmp_sparse_indices`时为Sliding Window Attention + Sparse Compressed Attention计算。

- 当`layout_q`为TND时，功能使用限制如下：
    - `q`的shape需要为[T1,N1,D]，其中N1仅支持64。
    - `ori_sparse_indices`的shape需要为[Q\_T, KV\_N, K1]，其中K1为对`ori_kv`一次离散选取的token数，K1仅支持512。
    - `cmp_sparse_indices`的shape需要为[Q\_T, KV\_N, K2]，其中K2为对`cmp_kv`一次离散选取的token数，K2仅支持512。
    - `cu_seqlens_q`必须传入，输入维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值。

- 当`layout_q`为BSND时，功能使用限制如下：
    - `q`的shape需要为[B, Q\_S,N1,D]，其中N1仅支持64。
    - `ori_sparse_indices`的shape需要为[B, Q\_S, KV\_N, K1]，其中K1为对`ori_kv`一次离散选取的token数，K1仅支持512。
    - `cmp_sparse_indices`的shape需要为[B, Q\_S, KV\_N, K2]，其中K2为对`cmp_kv`一次离散选取的token数，K2仅支持512。

- PageAttention场景下，功能使用限制如下：
    - `ori_kv`和`cmp_kv`的shape分别为[ori\_block\_num, ori\_block\_size, KV\_N, D]和[cmp\_block\_num, cmp\_block\_size, KV\_N, D]，其中ori\_block\_num和cmp\_block\_num为PageAttention时block总数，ori\_block\_size和cmp\_block\_size为一个block的token数，ori\_block\_size和cmp\_block\_size取值为16的倍数，最大支持1024，KV_N仅支持1。
    - `ori_block_table`和`cmp_block_table`的shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的S2和S3对应的block数量，即S2\_max / block\_size和S3\_max / block\_size向上取整。
- `metadata`为算子实际需要使用的分核结果，目前该参数必传，shape大小固定为[1024]。
- `layout_kv`仅支持输入PA_ND，故设置`cu_seqlens_ori_kv`和`cu_seqlens_cmp_kv`无效。
- 目前暂不支持返回`softmax_lse`，`return_softmax_lse`仅支持输入False，返回值`softmax_lse`为无效值。
- ori_mask_mode及cmp_mask_mode所表示的mask模式的详细介绍见[sparse_mode参数说明](../../../docs/zh/context/sparse_mode参数说明.md)。
- 目前暂不支持指定`q`中参与运算的token数，因此设置`seqused_q`无效。
- 目前暂不支持对`ori_kv`进行稀疏计算，因此设置`ori_sparse_indices`无效。
- 目前所有输入不支持传入空tensor。
- `q`、`ori_kv`、`cmp_kv`数据排布格式支持从多种维度解读，B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
- Q\_S和S1表示q shape中的S，S2表示ori_kv shape中的S，S3表示cmp_kv shape中的S；Q\_N和N1表示num\_q\_heads，KV\_N和N2表示num\_ori_kv\_heads和num\_cmp_kv\_heads；Q\_T和T1表示q shape中的输入样本序列长度的累加和。

- 当`layout_kv`为BSND时，功能使用限制如下：
    - `ori_kv`和`cmp_kv`的layout都必须为BSND，ori_kv的shape为[B, S2, N2,D]，cmp_kv的shape为[B, S3, N2,D]。

## Atlas A3 推理系列产品 调用说明

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    import random
    import math
    import custom_ops

    data_type = torch.bfloat16
    softmax_scale = 0.041666666666666664
    b = 4
    s1 = 128
    s2 = 8192
    n1 = 64
    n2 = 1
    dn = 512
    k = 512
    ori_block_size = 128
    cmp_block_size = 128
    s2_act = 4096
    cmp_ratio = 4
    ori_win_left = 127
    ori_win_right = 0
    layout_q = 'TND'
    layout_kv = 'PA_ND'
    ori_mask_mode = 4
    cmp_mask_mode = 3
    q = torch.tensor(np.random.uniform(-10, 10, (b*s1, n1, dn))).to(data_type).npu()

    cu_seqlens_q = torch.arange(0, (b + 1) * s1, step=s1).to(torch.int32).npu()
    t = cu_seqlens_q[-1].item()
    seqused_kv = torch.tensor([s2_act]*b).to(torch.int32).npu()

    cmp_kv_len = s2_act // cmp_ratio
    idxs = random.sample(range(cmp_kv_len - s1 + 1),  k)
    cmp_sparse_indices = torch.tensor([idxs for _ in range(t * n2)]).reshape(t, n2, k). \
        to(torch.int32).npu()

    ori_block_num =  math.ceil(s2_act/ori_block_size) * b
    ori_block_table = torch.tensor(np.random.permutation(range(ori_block_num))).to(torch.int32).reshape(b, -1).npu()
    ori_kv = torch.tensor(np.random.uniform(-5, 10, (ori_block_num, ori_block_size, n2, dn))).to(data_type).npu()

    block_num2 =  math.ceil(cmp_kv_len/ori_block_size) * b
    cmp_block_table = torch.tensor(np.random.permutation(range(block_num2))).to(torch.int32).reshape(b, -1).npu()
    cmp_kv = torch.tensor(np.random.uniform(-5, 10, (block_num2, cmp_block_size, n2, dn))).to(data_type).npu()
    sinks = torch.rand(n1).to(torch.float32).npu()
    metadata = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
        num_heads_q=n1,
        num_heads_kv=n2,
        head_dim=dn,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        batch_size=b,
        max_seqlen_q=s1,
        max_seqlen_kv=s2,
        cmp_topk=k,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=True,
        has_cmp_kv=True
    )
    attn_out, softmax_lse = torch.ops.custom.npu_sparse_attn_sharedkv(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=None,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_kv=seqused_kv,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=softmax_scale,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        return_softmax_lse=False)
    ```

- aclgraph模式调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    import random
    import math
    import torchair
    import custom_ops

    data_type = torch.bfloat16
    softmax_scale = 0.041666666666666664
    b = 4
    s1 = 128
    s2 = 8192
    n1 = 64
    n2 = 1
    dn = 512
    k = 512
    ori_block_size = 128
    cmp_block_size = 128
    s2_act = 4096
    cmp_ratio = 4
    ori_win_left = 127
    ori_win_right = 0
    layout_q = 'TND'
    layout_kv = 'PA_ND'
    ori_mask_mode = 4
    cmp_mask_mode = 3
    q = torch.tensor(np.random.uniform(-10, 10, (b*s1, n1, dn))).to(data_type).npu()

    cu_seqlens_q = torch.arange(0, (b + 1) * s1, step=s1).to(torch.int32).npu()
    t = cu_seqlens_q[-1].item()
    seqused_kv = torch.tensor([s2_act]*b).to(torch.int32).npu()

    cmp_kv_len = s2_act // cmp_ratio
    idxs = random.sample(range(cmp_kv_len - s1 + 1),  k)
    cmp_sparse_indices = torch.tensor([idxs for _ in range(t * n2)]).reshape(t, n2, k). \
        to(torch.int32).npu()

    ori_block_num =  math.ceil(s2_act/ori_block_size) * b
    ori_block_table = torch.tensor(np.random.permutation(range(ori_block_num))).to(torch.int32).reshape(b, -1).npu()
    ori_kv = torch.tensor(np.random.uniform(-5, 10, (ori_block_num, ori_block_size, n2, dn))).to(data_type).npu()

    block_num2 =  math.ceil(cmp_kv_len/ori_block_size) * b
    cmp_block_table = torch.tensor(np.random.permutation(range(block_num2))).to(torch.int32).reshape(b, -1).npu()
    cmp_kv = torch.tensor(np.random.uniform(-5, 10, (block_num2, cmp_block_size, n2, dn))).to(data_type).npu()
    sinks = torch.rand(n1).to(torch.float32).npu()

    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    class Network(torch.nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        def forward(self, num_heads_q, num_heads_kv, head_dim, batch_size, max_seqlen_q, max_seqlen_kv,
            topk, has_ori_kv, has_cmp_kv, q, ori_kv, cmp_kv, cmp_sparse_indices, ori_block_table,
            cmp_block_table, cu_seqlens_q, seqused_kv, softmax_scale, cmp_ratio, sinks,
            ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, layout_q, layout_kv):
            metadata = torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
                num_heads_q=num_heads_q,
                num_heads_kv=num_heads_kv,
                head_dim=head_dim,
                cu_seqlens_q=cu_seqlens_q,
                seqused_kv=seqused_kv,
                batch_size=batch_size,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                cmp_topk=topk,
                cmp_ratio=cmp_ratio,
                ori_mask_mode=ori_mask_mode,
                cmp_mask_mode=cmp_mask_mode,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
                layout_q=layout_q,
                layout_kv=layout_kv,
                has_ori_kv=has_ori_kv,
                has_cmp_kv=has_cmp_kv,
                device="npu:0"
            )
            npu_out = torch.ops.custom.npu_sparse_attn_sharedkv(
                q,
                ori_kv=ori_kv,
                cmp_kv=cmp_kv,
                ori_sparse_indices=None,
                cmp_sparse_indices=cmp_sparse_indices,
                ori_block_table=ori_block_table,
                cmp_block_table=cmp_block_table,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_ori_kv=None,
                cu_seqlens_cmp_kv=None,
                seqused_q=None,
                seqused_kv=seqused_kv,
                sinks=sinks,
                metadata=metadata,
                softmax_scale=softmax_scale,
                cmp_ratio=cmp_ratio,
                ori_mask_mode=ori_mask_mode,
                cmp_mask_mode=cmp_mask_mode,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
                layout_q=layout_q,
                layout_kv=layout_kv,
                return_softmax_lse=False)
            return npu_out

    mod = torch.compile(Network().npu(), backend=npu_backend, fullgraph=True)
    attn_out, softmax_lse = mod(
        num_heads_q=n1,
        num_heads_kv=n2,
        head_dim=dn,
        batch_size=b,
        max_seqlen_q=s1,
        max_seqlen_kv=s2,
        topk=k,
        has_ori_kv=True,
        has_cmp_kv=True,
        q=q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        softmax_scale=softmax_scale,
        cmp_ratio=cmp_ratio,
        sinks=sinks,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv)
    ```

更多使用示例见[pytest示例](./tests/pytest/README.md)。
