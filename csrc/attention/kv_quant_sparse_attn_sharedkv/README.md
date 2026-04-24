# KvQuantSparseAttnSharedkv

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>   | √  |

## 功能说明
- API功能：KvQuantSparseAttentionSharedKv 算子旨在完成以下公式描述的Attention计算，支持Sliding Window Attention、Compressed Attention以及Sparse Compressed Attention：

- 计算公式：

    $$
    O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
    $$

    其中$\tilde{K}=\tilde{V}$为基于入参控制的实际参与计算的$KV$。

## 函数原型

```
custom.npu_kv_quant_sparse_attn_sharedkv(q, kv_quant_mode, *, ori_kv=None, cmp_kv=None, ori_sparse_indices=None, cmp_sparse_indices=None, ori_block_table=None, cmp_block_table=None, cu_seqlens_q=None, cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None, seqused_q=None, seqused_kv=None, sinks=None, metadata=None, tile_size=0, rope_head_dim=0, softmax_scale=0, cmp_ratio=0, ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=127, ori_win_right=0, layout_q='BSND', layout_kv='PA_ND', return_softmax_lse=False) -> (Tensor, Tensor)
```

## 参数说明

> [!NOTE]  
>- q、ori\_kv、cmp\_kv参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Hidden Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
>- Q\_S和S1表示q shape中的S，S2表示ori\_kv shape中的S，S3表示cmp\_kv shape中的S；Q\_N和N1表示num\_q\_heads，KV\_N和N2表示num\_ori\_kv\_heads和num\_cmp\_kv\_heads；T1表示q shape中的T，T2表示ori\_kv shape中的T，T3表示cmp\_kv shape中的输入样本序列长度的累加和。
-   **q**（`Tensor`）：必选参数，对应公式中的$Q$，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`。`layout_query`为BSND时shape为[B, S1, N1, D]，当`layout_query`为TND时shape为[T1, N1, D]，其中N1仅支持64，D仅支持512。

-   **kv\_quant\_mode**（`int`）：必选参数，kv nope的量化模式，仅支持1，表示K、V nope为per-tile量化，量化后的KV数据类型为float8_e4m3。

- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

-   **ori\_kv**（`Tensor`）：可选参数，对应公式中的$\tilde{K}$和$\tilde{V}$的一部分，为原始不经压缩的KV，不支持非连续，数据格式支持ND，数据类型支持`float8_e4m3fn`，`layout_kv`为PA\_ND时shape为[block\_num1, block\_size1, KV\_N, D]，其中block\_num1为PageAttention时block总数，bloc\_size1为一个block的token数，block\_size1取值为16的倍数，最大支持1024，KV\_N仅支持1。D仅支持640，由ori\_kv\_nope、ori\_kv\_rope及量化参数在D方向拼接组成，并向上对齐128B。

-   **cmp\_kv**（`Tensor`）：可选参数，对应公式中的$\tilde{K}$和$\tilde{V}$的一部分，为经过压缩的KV，不支持非连续，数据格式支持ND，数据类型支持`float8_e4m3fn`，`layout_kv`为PA\_ND时shape为[block\_num2, block\_size, KV\_N, D]，其中block\_num2为PageAttention时block总数，block\_size2为一个block的token数，block\_size2取值为16的倍数，最大支持1024，KV\_N仅支持1。D仅支持640，由cmp\_kv\_nope、cmp\_kv\_rope及量化参数在D方向拼接组成，并向上对齐128B。

-   **ori\_sparse\_indices**（`Tensor`）：可选参数，代表离散取oriKvCache的索引，不支持非连续，数据格式支持ND,数据类型支持`int32`。当`layout_query`为BSND时，shape需要传入[B, Q\_S, KV\_N, K1]，其中K1为对`ori_kv`一次离散选取的block数，需要保证每行有效值均在前半部分，无效值均在后半部分，K1仅支持512。**目前暂不支持对ori_kv进行稀疏计算，故设置此参数无效。**

-   **cmp\_sparse\_indices**（`Tensor`）：可选参数，代表离散取cmpKvCache的索引，不支持非连续，数据格式支持ND,数据类型支持`int32`。当`layout_query`为BSND时，shape需要传入[B, Q\_S, KV\_N, K2]，其中K2为对`cmp_kv`一次离散选取的block数，需要保证每行有效值均在前半部分，无效值均在后半部分，K2仅支持512。

-   **ori\_block\_table**（`Tensor`）：可选参数，表示PageAttention中oriKvCache存储使用的block映射表。数据格式支持ND，数据类型支持`int32`，shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的S2对应的block数量，即S2\_max / block\_size向上取整。

-   **cmp\_block\_table**（`Tensor`）：可选参数，表示PageAttention中cmpKvCache存储使用的block映射表。数据格式支持ND，数据类型支持`int32`，shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的S3对应的block数量，即S3\_max / block\_size向上取整。

-   **cu\_seqlens\_q**（`Tensor`）：可选参数，当`layout_query`为TND时，表示不同Batch中`q`的有效token数，维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值，数据类型支持`int32`。

-   **cu\_seqlens\_ori\_kv**（`Tensor`）：可选参数，当`layout_kv`为TND时，表示不同Batch中`ori_kv`的有效token数，维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值，数据类型支持`int32`。**目前layout_kv仅支持PA_ND，故设置此参数无效。**

-   **cu\_seqlens\_cmp\_kv**（`Tensor`）：可选参数，当`layout_kv`为TND时，表示不同Batch中`cmp_kv`的有效token数，维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值，数据类型支持`int32`。**目前layout_kv仅支持PA_ND，故设置此参数无效。**

-   **seqused\_q**（`Tensor`）：可选参数，表示不同Batch中`q`实际参与运算的token数，维度为B，数据格式支持ND，数据类型支持`int32`，不输入则所有token均参与运算。**目前暂不支持指定该参数。**

-   **seqused\_kv**（`Tensor`）：可选参数，表示不同Batch中`ori_kv`实际参与运算的token数，维度为B，数据格式支持ND，数据类型支持`int32`，不输入则所有token均参与运算。

-   **sinks**（`Tensor`）：可选参数，注意力下沉tensor，数据格式支持ND，数据类型支持`float32`，shape为[N1]。

-   **metadata**（`Tensor`）：可选参数，为aicpu算子（kv_quant_npu_sparse_attn_sharedkv_metadata）的分核结果，数据格式支持ND，数据类型支持`int32`，shape固定为[2048]。

-   **tile\_size**（`int`）：可选参数，表示量化粒度，必须能被rope_head_dim整除，默认值为None，当前仅支持64。

-   **rope\_head\_dim**（`int`）：可选参数，数据类型支持`int32`，默认值为0，当前仅支持64。

-   **softmax\_scale**（`float`）：可选参数，代表缩放系数，作为q与ori_kv和cmp_kv矩阵乘后Muls的scalar值，数据类型支持`float`，默认值为None，None表示softmax_scale值为1/sqrt(512)。
    
-   **cmp\_ratio**（`int`）：可选参数，表示对ori_kv的压缩率，数据类型支持`int`，数据范围支持4/128，默认值为None。

-   **ori\_mask\_mode**（`int`）：可选参数，表示q和ori_kv计算的mask模式，仅支持输入默认值4，代表band模式的mask，数据类型支持`int`。

-   **cmp\_mask\_mode**（`int`）：可选参数，表示q和cmp_kv计算的mask模式，仅支持输入默认值3，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景，数据类型支持`int`。

-   **ori\_win\_left**（`int`）：可选参数，表示q和ori_kv计算中q对过去token计算的数量，数据类型支持`int`，仅支持默认值127。

-   **ori\_win\_right**（`int`）：可选参数，表示q和ori_kv计算中q对未来token计算的数量，数据类型支持`int`，仅支持默认值0。

-   **layout\_q**（`str`）：可选参数，用于标识输入q的数据排布格式，默认值为BSND，支持传入BSND和TND。

-   **layout\_kv**（`str`）：可选参数，用于标识输入`ori_kv`和`cmp_kv`的数据排布格式，仅支持传入默认值PA_ND（PageAttention）。

-   **return\_softmax\_lse**（`bool`）：可选参数，表示是否输出softmax\_lse。True表示返回，False表示不返回；默认值为False。**目前暂不支持返回softmax_lse。**

## 返回值说明

-   **attention\_out**（`Tensor`）：公式中的输出。数据格式支持ND，数据类型支持`bfloat16`和。当layout\_q为BSND时shape为[B, S1, N1, D]，当layout\_q为TND时shape为[T1, N1, D]。
-   **softmax\_lse**（`Tensor`）：可选输出，输出q乘ori\_kv的结果先取max得到softmax\_max，query乘key的结果减去softmax\_max，再取exp，最后取sum，得到softmax\_sum，最后对softmax\_sum取log，再加上softmax\_max得到的结果。数据类型支持`float`。当layout\_q为BSND时shape为[B, N2, S1, N1/N2]，当layout\_q为TND时shape为[N2, T1, N1/N2]。**目前softmax_lse输出为无效值。**

## 约束说明

-   该接口支持推理场景下使用。
-   该接口支持aclgraph模式。
-   参数q中的D仅支持512，ori\_kv、cmp\_kv的D值仅支持640，由kv\_nope、kv\_rope及量化参数在D方向拼接组成，并向上对齐128B。
-   参数ori\_kv、cmp\_kv的数据类型必须保持一致。

## 调用示例
- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import numpy as np
    import random
    import math
    import custom_ops

    layout_q="TND"
    layout_kv="PA_ND"
    q_type=torch.bfloat16
    ori_kv_type=torch.float8_e4m3fn
    cmp_kv_type=torch.float8_e4m3fn
    B = 1
    S1 = 1
    T1 = 1
    S2 = 8193
    actS2 = 8193
    N1 = 64
    N2 = 1
    D = 512
    K = 512
    ori_block_size = 128
    cmp_block_size = 128
    softmax_scale = 0.04419417
    cmp_ratio = 4
    ori_mask_mode = 4
    cmp_mask_mode = 3
    ori_win_left = 127
    ori_win_right = 0
    kv_quant_mode = 1
    tile_size = 64
    rope_head_dim = 64

    quant_scale_head_dim = (D + tile_size - 1) // tile_size
    d_aligned_128 = (D + rope_head_dim * 2 + quant_scale_head_dim + 127) // 128 *128

    q = torch.tensor(np.random.uniform(-10, 10, (B*S1, N1, D))).to(q_type).npu()
        
    cu_seqlens_q = torch.arange(0, (B + 1) * S1, step=S1).to(torch.int32).npu()
    seqused_kv = torch.tensor([S2]*B).to(torch.int32).npu()

    cmp_kv_len = actS2 // cmp_ratio
    idxs = random.sample(range(cmp_kv_len - S1 + 1),  K)
    cmp_sparse_indices = torch.tensor([idxs for _ in range(B * S1 * N2)]).reshape(B, S1, N2, K).to(torch.int32).npu()
        
    ori_block_num =  math.ceil(actS2/ori_block_size) * B
    block_table1 = torch.tensor(np.random.permutation(range(ori_block_num))).to(torch.int32).reshape(B, -1).npu()
    ori_kv = torch.tensor(np.random.uniform(-5, 10, (ori_block_num, ori_block_size, N2, D))).to(ori_kv_type).npu()

    cmp_block_num =  math.ceil(cmp_kv_len/cmp_block_size) * B
    block_table2 = torch.tensor(np.random.permutation(range(cmp_block_num))).to(torch.int32).reshape(B, -1).npu()
    cmp_kv = torch.tensor(np.random.uniform(-5, 10, (cmp_block_num, cmp_block_size, N2, D))).to(cmp_kv_type).npu()
    sinks = torch.rand(N1).to(torch.float32).npu()

    metadata = torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv_metadata(
        num_heads_q=N1,
        num_heads_kv=N2,
        head_dim=D,
        kv_quant_mode=1,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=torch.tensor([]).npu(),
        cu_seqlens_cmp_kv=torch.tensor([]).npu(),
        seqused_q=torch.tensor([]).npu(),
        seqused_kv=seqused_kv,
        batch_size=B,
        max_seqlen_q=S1,
        max_seqlen_kv=S2,
        cmp_topk=K,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=True,
        has_cmp_kv=True,
        device = "npu:0")

    attn_out = torch_npu.npu_kv_quant_sparse_attn_sharedkv(
        q=q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=block_table1,
        cmp_block_table=block_table2,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        sinks=sinks,
        metadata=metadata,
        kv_quant_mode=kv_quant_mode,
        tile_size=tile_size,
        rope_head_dim=rope_head_dim,
        softmax_scale=softmax_scale,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv)
    ``` 

-   图模式调用
    ```python
    import torch
    import torch_npu
    import numpy as np
    import random
    import math
    import custom_ops
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    class Network(torch.nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        def forward(self, B, N1, N2, D, K, S1, S2, q, ori_kv, cmp_kv, cmp_sparse_indices, ori_block_table, 
            cmp_block_table, cu_seqlens_q, seqused_kv, sinks, kv_quant_mode, tile_size, rope_head_dim, 
            softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, layout_q, layout_kv):
            metadata = torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv_metadata(
                num_heads_q=N1,
                num_heads_kv=N2,
                head_dim=D,
                kv_quant_mode=1,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_ori_kv=torch.tensor([]).npu(),
                cu_seqlens_cmp_kv=torch.tensor([]).npu(),
                seqused_q=torch.tensor([]).npu(),
                seqused_kv=seqused_kv,
                batch_size=B,
                max_seqlen_q=S1,
                max_seqlen_kv=S2,
                cmp_topk=K,
                cmp_ratio=cmp_ratio,
                ori_mask_mode=ori_mask_mode,
                cmp_mask_mode=cmp_mask_mode,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
                layout_q=layout_q,
                layout_kv=layout_kv,
                has_ori_kv=True,
                has_cmp_kv=True,
                device = "npu:0")

            npu_out = torch_npu.npu_kv_quant_sparse_attn_sharedkv(
                q=q,
                ori_kv=ori_kv,
                cmp_kv=cmp_kv,
                cmp_sparse_indices=cmp_sparse_indices,
                ori_block_table=ori_block_table,
                cmp_block_table=cmp_block_table,
                cu_seqlens_q=cu_seqlens_q,
                seqused_kv=seqused_kv,
                sinks=sinks,
                metadata=metadata,
                kv_quant_mode=kv_quant_mode,
                tile_size=tile_size,
                rope_head_dim=rope_head_dim,
                softmax_scale=softmax_scale,
                cmp_ratio=cmp_ratio,
                ori_mask_mode=ori_mask_mode,
                cmp_mask_mode=cmp_mask_mode,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
                layout_q=layout_q,
                layout_kv=layout_kv)
            return npu_out

    layout_q="TND"
    layout_kv="PA_ND"
    q_type=torch.bfloat16
    ori_kv_type=torch.float8_e4m3fn
    cmp_kv_type=torch.float8_e4m3fn
    B = 1
    S1 = 1
    T1 = 1
    S2 = 8193
    actS2 = 8193
    N1 = 64
    N2 = 1
    D = 512
    K = 512
    ori_block_size = 128
    cmp_block_size = 128
    softmax_scale = 0.04419417
    cmp_ratio = 4
    ori_mask_mode = 4
    cmp_mask_mode = 3
    ori_win_left = 127
    ori_win_right = 0
    kv_quant_mode = 1
    tile_size = 64
    rope_head_dim = 64

    quant_scale_head_dim = (D + tile_size - 1) // tile_size
    d_aligned_128 = (D + rope_head_dim * 2 + quant_scale_head_dim + 127) // 128 *128

    q = torch.tensor(np.random.uniform(-10, 10, (B*S1, N1, D))).to(q_type).npu()
        
    cu_seqlens_q = torch.arange(0, (B + 1) * S1, step=S1).to(torch.int32).npu()
    seqused_kv = torch.tensor([S2]*B).to(torch.int32).npu()

    cmp_kv_len = actS2 // cmp_ratio
    idxs = random.sample(range(cmp_kv_len - S1 + 1),  K)
    cmp_sparse_indices = torch.tensor([idxs for _ in range(B * S1 * N2)]).reshape(B, S1, N2, K).to(torch.int32).npu()
        
    ori_block_num =  math.ceil(actS2/ori_block_size) * B
    block_table1 = torch.tensor(np.random.permutation(range(ori_block_num))).to(torch.int32).reshape(B, -1).npu()
    ori_kv = torch.tensor(np.random.uniform(-5, 10, (ori_block_num, ori_block_size, N2, D))).to(ori_kv_type).npu()

    cmp_block_num =  math.ceil(cmp_kv_len/cmp_block_size) * B
    block_table2 = torch.tensor(np.random.permutation(range(cmp_block_num))).to(torch.int32).reshape(B, -1).npu()
    cmp_kv = torch.tensor(np.random.uniform(-5, 10, (cmp_block_num, cmp_block_size, N2, D))).to(cmp_kv_type).npu()
    sinks = torch.rand(N1).to(torch.float32).npu()

    torch._dynamo.reset()
    npu_mode = Network().npu()
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)

    attn_out = npu_mode(
                B=B,
                N1=N1,
                N2=N2,
                D=D,
                K=K,
                S1=S1,
                S2=S2,
                q=q,
                ori_kv=ori_kv,
                cmp_kv=cmp_kv,
                cmp_sparse_indices=cmp_sparse_indices,
                ori_block_table=block_table1,
                cmp_block_table=block_table2,
                cu_seqlens_q=cu_seqlens_q,
                seqused_kv=seqused_kv,
                sinks=sinks,
                kv_quant_mode=kv_quant_mode,
                tile_size=tile_size,
                rope_head_dim=rope_head_dim,
                softmax_scale=softmax_scale,
                cmp_ratio=cmp_ratio,
                ori_mask_mode=ori_mask_mode,
                cmp_mask_mode=cmp_mask_mode,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
                layout_q=layout_q,
                layout_kv=layout_kv)
    ```