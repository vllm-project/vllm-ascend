# generic_block_sparse_attention

A5新增TND非PA代码路径（尚未运行构建或设备验证）：`layout_q="TND"`、`layout_kv="TND"`，Q为`[Tq,Nq,128]`，K/V为`[Tkv,Nkv,128]`，仅支持float16/bfloat16、`quant_mode=0`。Q/KV各自必传`cu_seqlens_*`，`seqused_*`可选，`block_table=None`。Metadata调用须使用相同布局及长度输入；支持`return_softmax_lse=True`和既有FD调度。

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：

  `generic_block_sparse_attention`是基于`TorchNPU`的`cann_ops_transformer`扩展接口，用于调用`GenericBlockSparseAttention`算子完成任意粒度块稀疏注意力计算。

  `generic_block_sparse_attention_metadata`是`generic_block_sparse_attention`的元数据生成接口，用于在主算子执行前生成metadata。metadata记录AICore/AIVCore的任务切分结果，主算子可选择传入该metadata以优化调度。典型调用流程如下：

  1. 准备`q`、`k`、`v`、`sparse_block_idx`、`sparse_block_count`等输入。
  2. 调用`generic_block_sparse_attention_metadata`生成`metadata`。
  3. 调用`generic_block_sparse_attention`，将上一步得到的`metadata`传入主算子。

- **计算公式**：

  本算子在标准 Attention 基础上，沿序列维按块进行稀疏计算。稀疏块大小为$blockShapeX \times blockShapeY$：Q 按$blockShapeX$、KV 按$blockShapeY$划分；`sparse_block_idx`给出每个Q块选中的KV块索引，`sparse_block_count`给出每个Q块实际参与计算的KV块个数，仅对这些块完成$q k^{T}$、Softmax及与$v$的乘积。

  $$
  attention\_out = Softmax(softmax\_scale \cdot q \cdot k_{sparse}^{T} + attn\_mask) \cdot v_{sparse}
  $$

  其中$softmax\_scale$为缩放系数；$k_{sparse}$、$v_{sparse}$表示按上述索引选取后的KV块。

> [!NOTE]
>
> q、k、v数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小batch_size、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸headdim，且满足D=H/N、Q_T表示所有query Batch输入样本序列长度的累加和，KV_T表示所有k、v Batch输入样本序列长度的累加和。
> Q_S表示seqlen_q，Q_N表示nheads_q，KV_S表示seqlen_kv，KV_N表示nheads_kv。

## 函数原型

调用generic_block_sparse_attention接口之前，请先调用前置接口generic_block_sparse_attention_metadata，完成generic_block_sparse_attention负载均衡的计算。

```python
cann_ops_transformer.generic_block_sparse_attention_metadata(
    sparse_block_idx,
    sparse_block_count,
    num_heads_q,
    num_heads_kv,
    head_dim,
    block_shape,
    *,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    is_packed_gqa=True,
    layout_q="TND",
    layout_kv="PA_BBND",
    mask_mode=1,
    quant_mode=0,
    softmax_precision=1,
    win_left=-1,
    win_right=-1,
) -> Tensor
```

```python
cann_ops_transformer.generic_block_sparse_attention(
    q,
    k,
    v,
    sparse_block_idx,
    sparse_block_count,
    block_shape,
    *,
    metadata=None,
    attn_mask=None,
    q_dequant_scale=None,
    k_dequant_scale=None,
    v_dequant_scale=None,
    p_quant_scale=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    block_table=None,
    is_packed_gqa=True,
    layout_q="TND",
    layout_kv="PA_BBND",
    softmax_scale=0.0,
    mask_mode=1,
    quant_mode=0,
    dst_type_max=0.0,
    softmax_precision=1,
    win_left=-1,
    win_right=-1,
    return_softmax_lse=False,
    attention_out_dtype=None,
) -> (Tensor, Tensor)
```

## 枚举说明

`quant_mode` 与 `mask_mode` 在 Python 接口中支持传入 `IntEnum` 枚举或对应 int 值，枚举定义于 `cann_ops_transformer.ops.generic_block_sparse_attention`：

### quant_mode 枚举

| 枚举名 | 值 | 含义 |
| :--- | :---: | :--- |
| `NO_QUANT` | 0 | 非量化（默认值） |
| `FP8_E4M3_STATIC_PER_GROUP` | 1 | FP8_E4M3 静态 per-group |
| `FP8_E4M3_DYNAMIC_MX` | 2 | FP8_E4M3 动态 MX |
| `FP4_E2M1_DYNAMIC_OCP` | 3 | FP4_E2M1 动态 OCP |
| `FP4_E2M1_DYNAMIC_CX` | 4 | FP4_E2M1 动态 CX |
| `FP8_E4M3_STATIC_CAST_P` | 5 | FP8_E4M3 静态 cast P |

### mask_mode 枚举

| 枚举名 | 值 | 含义 |
| :--- | :---: | :--- |
| `NO_MASK` | 0 | 不加 mask |
| `CAUSAL` | 1 | Causal 模式（默认值） |
| `WINDOW` | 2 | Window 模式 |

> [!NOTE]
>
> 枚举为 `IntEnum`，可直接作为 int 传入底层算子；接口仅支持传入枚举或对应 int 值。当前仅支持 mask_mode = 1（`CAUSAL`）；quant_mode 当前仅支持 0（`NO_QUANT`）与 5（`FP8_E4M3_STATIC_CAST_P`）。

## 参数说明

### generic_block_sparse_attention_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| sparse_block_idx | Tensor | 必选 | 每个Q块选择的KV块索引 | int32 | ND | (KV_N, totalQBlocks, topK) |
| sparse_block_count | Tensor | 必选 | 每个Q块实际保留的KV块数量 | int32 | ND | (KV_N, totalQBlocks) |
| num_heads_q | int | 必选 | Query head数 | int32 | - | - |
| num_heads_kv | int | 必选 | Key/Value head数 | int32 | - | - |
| head_dim | int | 必选 | 每个注意力头的维度 | int32 | - | - |
| block_shape | list[int] | 必选 | 稀疏块形状 `[block_x, block_y]` | int64 | - | 长度为2 |
| cu_seqlens_q | Tensor | 可选 | 累积序列长度，用于处理变长序列，第一个元素必须为0 | int64 | ND | (B+1,) |
| cu_seqlens_kv | Tensor | 可选 | 累积序列长度，用于处理变长序列，第一个元素必须为0 | int64 | ND | (B+1,) |
| seqused_q | Tensor | 可选 | 指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (B,) |
| seqused_kv | Tensor | 可选 | 指定每batch中实际使用的kv序列长度，截断冗余运算 | int32 | ND | (B,) |
| max_seqlen_q | int | 可选 | 指定查询q序列的长度上限 | int32 | - | - |
| max_seqlen_kv | int | 可选 | 指定键k和值v序列的长度上限 | int32 | - | - |
| is_packed_gqa | bool | 可选 | 是否启用Packed GQA | bool | - | - |
| layout_q | string | 可选 | 定义输入query张量的布局格式 | string | - | - |
| layout_kv | string | 可选 | 定义输入key/value张量的布局格式 | string | - | - |
| mask_mode | int/MaskMode | 可选 | 掩码模式，支持传入枚举或对应 int 值，枚举定义见「mask_mode 枚举」 | int32 | - | - |
| quant_mode | int/QuantMode | 可选 | 量化模式，支持传入枚举或对应 int 值，枚举定义见「quant_mode 枚举」 | int32 | - | - |
| softmax_precision | int | 可选 | Softmax精度模式 | int32 | - | - |
| win_left | int | 可选 | window左界限 | int32 | - | - |
| win_right | int | 可选 | window右界限 | int32 | - | - |

### generic_block_sparse_attention

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | 必选 | 公式中的q | bfloat16/float16/float8_e4m3fn | ND | (Q_T, Q_N, D) |
| k | Tensor | 必选 | 公式中的k | bfloat16/float16/float8_e4m3fn | ND | PA: (num_blocks, block_size, KV_N, D)；A5 TND非PA: (KV_T, KV_N, D) |
| v | Tensor | 必选 | 公式中的v | bfloat16/float16/float8_e4m3fn | ND | PA: (num_blocks, block_size, KV_N, D)；A5 TND非PA: (KV_T, KV_N, D) |
| sparse_block_idx | Tensor | 必选 | 每个Q块选择的KV块索引 | int32 | ND | (KV_N, totalQBlocks, topK) |
| sparse_block_count | Tensor | 必选 | 每个Q块实际保留的KV块数量 | int32 | ND | (KV_N, totalQBlocks) |
| block_shape | list[int] | 必选 | 稀疏块形状 `[block_x, block_y]` | int64 | - | 长度为2 |
| metadata | Tensor | 可选 | `generic_block_sparse_attention_metadata`生成的任务切分结果，传入后可优化调度 | int32 | ND | (1024,) |
| attn_mask | Tensor | 可选 | 掩码矩阵 | bool | ND | - |
| q_dequant_scale | Tensor | 可选 | query反量化缩放因子 | float32/float8_e8m0 | ND | - |
| k_dequant_scale | Tensor | 可选 | key反量化缩放因子 | float32/float8_e8m0 | ND | - |
| v_dequant_scale | Tensor | 可选 | value反量化缩放因子 | float32/float8_e8m0 | ND | - |
| p_quant_scale | Tensor | 可选 | P量化缩放因子 | float32 | ND | - |
| cu_seqlens_q | Tensor | 可选 | 累积序列长度，用于处理变长序列，第一个元素必须为0 | int64 | ND | (B+1,) |
| cu_seqlens_kv | Tensor | 可选 | 累积序列长度，用于处理变长序列，第一个元素必须为0 | int64 | ND | (B+1,) |
| seqused_q | Tensor | 可选 | 指定每batch中实际使用的序列长度，截断冗余运算 | int32 | ND | (B,) |
| seqused_kv | Tensor | 可选 | 指定每batch中实际使用的kv序列长度，截断冗余运算 | int32 | ND | (B,) |
| block_table | Tensor | 可选 | 用于Paged Attention计算中的块索引映射 | int32 | ND | (B, max_num_blocks_per_seq) |
| is_packed_gqa | bool | 可选 | 是否启用Packed GQA | bool | - | - |
| layout_q | string | 可选 | 定义输入query张量的布局格式 | string | - | - |
| layout_kv | string | 可选 | 定义输入key/value张量的布局格式 | string | - | - |
| softmax_scale | float | 可选 | 可显式设置缩放因子，覆盖默认计算 | float32 | - | - |
| mask_mode | int/MaskMode | 可选 | 掩码模式，支持传入枚举或对应 int 值，枚举定义见「mask_mode 枚举」 | int32 | - | - |
| quant_mode | int/QuantMode | 可选 | 量化模式，支持传入枚举或对应 int 值，枚举定义见「quant_mode 枚举」 | int32 | - | - |
| dst_type_max | float | 可选 | 量化相关参数 | float32 | - | - |
| softmax_precision | int | 可选 | Softmax精度模式 | int32 | - | - |
| win_left | int | 可选 | window左界限 | int32 | - | - |
| win_right | int | 可选 | window右界限 | int32 | - | - |
| return_softmax_lse | bool | 可选 | 是否需要获取softmax的LSE结果 | bool | - | - |
| attention_out_dtype | dtype | 可选 | 输出dtype；`quant_mode!=0`时必填，`quant_mode=0`且未指定时与q一致 | ScalarType | - | - |

## 返回值说明

### generic_block_sparse_attention_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | 必选 | generic_block_sparse_attention的任务切分数据 | int32 | ND | (1024,) |

### generic_block_sparse_attention

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| attention_out | Tensor | 必选 | generic_block_sparse_attention的计算输出 | bfloat16/float16 | ND | (Q_T, Q_N, D) |
| softmax_lse | Tensor | 可选 | softmax的LSE结果 | float32 | ND | (Q_T, Q_N, 1) 或空 |

**说明**

- attention_out：限制：该输出参数的shape与入参query的shape保持一致；`quant_mode=0`时dtype默认与q一致，`quant_mode!=0`时由attention_out_dtype指定。
- softmax_lse：return_softmax_lse为True时，TND布局下输出shape为(Q_T, Q_N, 1)的Tensor；return_softmax_lse为False时，则输出空Tensor。

## 约束说明

- 参数cu_seqlens_q、cu_seqlens_kv、seqused_q、seqused_kv、sparse_block_idx、sparse_block_count及block_table属于tensor。由于算子在Tiling阶段无法获取tensor的具体数值，tiling侧不对值进行校验，正确性需要用户自行保证。若上述参数传入非法值，会触发未定义行为（精度问题、非法内存访问导致的程序崩溃等）。
- generic_block_sparse_attention_metadata和generic_block_sparse_attention的入参在调用时应该保持一致。由于算子分为两个接口分段调用，算子无法自行校验，正确性需要由客户自行保证。若接口传入参数不一致，会发生未定义行为（精度问题、非法内存访问导致的程序崩溃等）。

### 特性参数组

|      特性参数组      |     参数字段名称     |    字段分组    |  字段类型  |
| :-------------------: | :-------------------: | :-------------: | :--------: |
|      公共参数组      |         q         |      INPUT      |   Tensor   |
|                      |         k         |      INPUT      |   Tensor   |
|                      |         v         |      INPUT      |   Tensor   |
|                      |  sparse_block_idx  |      INPUT      |   Tensor   |
|                      | sparse_block_count  |      INPUT      |   Tensor   |
|                      |     block_shape     |   ATTR(REQUIRED) |  int[]  |
|                      |    is_packed_gqa    | ATTR(OPTIONAL) |   bool   |
|                      |       metadata       | INPUT(OPTIONAL) |   Tensor   |
|                      |    softmax_scale    | ATTR(OPTIONAL) |   float   |
|                      |  softmax_precision  | ATTR(OPTIONAL) |   int   |
|                      |      layout_q       | ATTR(OPTIONAL) |   string   |
|                      |      layout_kv      | ATTR(OPTIONAL) |   string   |
|                      |   attention_out   |     OUTPUT     |   Tensor   |
|   metadata参数组   |   max_seqlen_q   | ATTR(OPTIONAL) |   int   |
|                      |  max_seqlen_kv  | ATTR(OPTIONAL) |   int   |
|                      |   num_heads_q   | ATTR(REQUIRED) |   int   |
|                      |  num_heads_kv  | ATTR(REQUIRED) |   int   |
|                      |     head_dim     | ATTR(REQUIRED) |   int   |
|                      |       metadata       |     OUTPUT     |   Tensor   |
|      Mask参数组      |      mask_mode      | ATTR(OPTIONAL) |   int   |
|                      |      win_left       | ATTR(OPTIONAL) |   int   |
|                      |     win_right      | ATTR(OPTIONAL) |   int   |
|                      |      attn_mask      | INPUT(OPTIONAL) |   Tensor   |
| SeqLens参数组  |   cu_seqlens_q   | INPUT(OPTIONAL) |  Tensor  |
|                      |  cu_seqlens_kv  | INPUT(OPTIONAL) |  Tensor  |
|                      |    seqused_q    | INPUT(OPTIONAL) |  Tensor  |
|                      |   seqused_kv   | INPUT(OPTIONAL) |  Tensor  |
| Paged Attention参数组 |     block_table     | INPUT(OPTIONAL) |   Tensor   |
|    Quant参数组    |    quant_mode    | ATTR(OPTIONAL) |   int   |
|                      | q_dequant_scale | INPUT(OPTIONAL) |  Tensor  |
|                      | k_dequant_scale | INPUT(OPTIONAL) |  Tensor  |
|                      | v_dequant_scale | INPUT(OPTIONAL) |  Tensor  |
|                      |  p_quant_scale  | INPUT(OPTIONAL) |  Tensor  |
|                      |  dst_type_max  | ATTR(OPTIONAL) |  float  |
|                      | attention_out_dtype | ATTR(OPTIONAL) | ScalarType |
|   SoftmaxLSE参数组   | return_softmax_lse | ATTR(OPTIONAL) |    bool    |
|                      |    softmax_lse    |     OUTPUT     |   Tensor   |

### 基准信息说明

资料约束中，常见字段释义如下：

|    命名    |                            含义                            |
| :---------: | :---------------------------------------------------------: |
|      B      |                Batch,表示输入样本批量大小                |
|     Q_N     |        输入q tensor的头数，对应q shape中的N        |
|    KV_N    |    输入k/v tensor的头数，对应k/v shape中的N    |
|     Q_T     |          输入q tensor所有batch序列长度的累加和          |
|    KV_T    |          输入k/v所有batch序列长度的累加和          |
|     Q_S     |      各batch的query逻辑序列长度，TND下一般由cu_seqlens_q差分得到      |
|    KV_S    |  各batch的key/value逻辑序列长度；TND下一般由cu_seqlens_kv差分得到，Paged Attention下为block_table映射对应的逻辑KV长度  |
|     D     |          输入q/k/v tensor隐藏层最小的单元尺寸headdim         |
|  topK  | sparse_block_idx最后一维，表示每个Q块最多选择的KV块数 |
| totalQBlocks | TND下各batch按Q_S分块后的Q块总数；$\sum_i\mathrm{ceil}(Q\_S_i / block\_x)$，为`sparse_block_idx`/`sparse_block_count`第二维 |
| num_blocks | Paged KV Cache物理页总数 |
| block_size | Paged KV Cache单页token数 |

### 参数组约束

#### 公共参数组

- 入参为空的场景处理：
  - 空Tensor指必选输入和输出的shape size为0,即有任意轴为0。
  - 触发空tensor的用例将全部拦截报错。

- q、k、v、attention_out校验:

    <table style="undefined;table-layout: fixed; width:1625px"><colgroup>
    <col style="width: 147px">
    <col style="width: 232px">
    <col style="width: 232px">
    <col style="width: 293px">
    <col style="width: 185px">
    </colgroup>
    <thead>
    <tr>
    <th>参数</th>
    <th>单参数校验</th>
    <th>存在性校验</th>
    <th>一致性校验</th>
    <th>特性交叉校验</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>q</td>
        <td>
            <ul>
                <li>tensor_type支持bfloat16、float16、float8_e4m3fn</li>
                <li>TND -> (Q_T, Q_N, D)</li>
            </ul>
        </td>
        <td rowspan="4">
            必须存在
        </td>
        <td rowspan="4">
            <ul>
                <li>q、k、v的数据类型需相同</li>
                <li>k、v的shape需相同</li>
                <li>Layout校验规则见<a href="#layout匹配关系表">layout匹配关系表</a></li>
            </ul>
        </td>
        <td rowspan="4">
            轴校验：
            <ul>
                <li>Q_T > 0</li>
                <li>Q_N > 0</li>
                <li>KV_N > 0</li>
                <li>当前D仅支持128</li>
                <li>Q_N % KV_N == 0且Q_N / KV_N > 0</li>
                <li>groupSize = Q_N / KV_N 当前须不超过128</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>k</td>
        <td rowspan="2">
            <ul>
                <li>tensor_type支持bfloat16、float16、float8_e4m3fn</li>
                <li>PA_BBND -> (num_blocks, block_size, KV_N, D)</li>
                <li>PA的block_size须等于128且等于block_shape[1]；A5 TND非PA的K/V为(KV_T, KV_N, D)</li>
                <li>非连续Tensor约束：PA_BBND仅dim0支持非连续</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>v</td>
    </tr>
    <tr>
        <td>attention_out</td>
        <td>
            <ul>
                <li>quant_mode=0时data_type与q一致；quant_mode!=0时由attention_out_dtype指定</li>
                <li>shape与q一致</li>
            </ul>
        </td>
    </tr>
    </tbody>
    </table>

- layout匹配关系表：<a name="layout匹配关系表"></a>

    <table style="undefined;table-layout: fixed; width:1625px"><colgroup>
    <col style="width: 247px">
    <col style="width: 232px">
    <col style="width: 293px">
    <col style="width: 293px">
    </colgroup>
    <thead>
    <tr>
        <th>layout_q</th>
        <th>layout_kv</th>
        <th>attention_out</th>
        <th>softmax_lse</th>
    </tr>
    </thead>
    <tbody>
        <tr>
            <td>TND</td>
            <td>PA_BBND；A5 quant_mode=0新增TND</td>
            <td>(Q_T, Q_N, D)</td>
            <td>(Q_T, Q_N, 1)</td>
        </tr>
    </tbody>
    </table>

- sparse_block_idx、sparse_block_count、block_shape、is_packed_gqa校验:

    <table style="undefined;table-layout: fixed; width:1625px"><colgroup>
    <col style="width: 147px">
    <col style="width: 232px">
    <col style="width: 232px">
    <col style="width: 293px">
    <col style="width: 185px">
    </colgroup>
    <thead>
    <tr>
    <th>参数</th>
    <th>单参数校验</th>
    <th>存在性校验</th>
    <th>一致性校验</th>
    <th>特性交叉校验</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>sparse_block_idx</td>
        <td>
            <ul>
                <li>tensor_type仅支持int32</li>
                <li>shape为(KV_N, totalQBlocks, topK)</li>
            </ul>
        </td>
        <td rowspan="3">必须存在</td>
        <td rowspan="3">
            <ul>
                <li>metadata接口与主算子须传入相同的sparse_block_idx、sparse_block_count、block_shape</li>
                <li>topK须不小于sparse_block_count中所有元素的最大值</li>
            </ul>
        </td>
        <td rowspan="3">
            <ul>
                <li>当前topK不超过256</li>
                <li>Arch35非量化FP16/BF16支持block_shape=[1, Y]，Y为1～256任意整数；其他场景为[1,128]。PA page size和attention KV tile均固定128。</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>sparse_block_count</td>
        <td>
            <ul>
                <li>tensor_type仅支持int32</li>
                <li>shape为(KV_N, totalQBlocks)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>block_shape</td>
        <td>
            <ul>
                <li>长度为2的int列表</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>is_packed_gqa</td>
        <td>
            <ul>
                <li>data_type仅支持BOOL</li>
                <li>当前仅支持True</li>
            </ul>
        </td>
        <td>可选属性，默认值为True</td>
        <td>metadata接口与主算子须一致</td>
        <td>无</td>
    </tr>
    </tbody>
    </table>

- metadata校验:

    <table style="undefined;table-layout: fixed; width:1625px">
        <colgroup>
            <col style="width: 147px">
            <col style="width: 232px">
            <col style="width: 232px">
            <col style="width: 293px">
            <col style="width: 185px">
        </colgroup>
        <thead>
            <tr>
                <th>参数</th>
                <th>单参数校验</th>
                <th>存在性校验</th>
                <th>一致性校验</th>
                <th>特性交叉校验</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>metadata</td>
                <td>
                    <ul>
                        <li>tensor_type仅支持int32</li>
                        <li>shape固定为(1024,)</li>
                        <li>当前不支持不传入，未传入将发出拦截报警</li>
                    </ul>
                </td>
                <td>可选参数</td>
                <td>无</td>
                <td>传入时需与generic_block_sparse_attention_metadata生成的结果一致</td>
            </tr>
        </tbody>
    </table>

- softmax_precision校验:

    <table style="undefined;table-layout: fixed; width:1625px">
        <colgroup>
            <col style="width: 147px">
            <col style="width: 232px">
            <col style="width: 232px">
            <col style="width: 293px">
            <col style="width: 185px">
        </colgroup>
        <thead>
            <tr>
                <th>参数</th>
                <th>单参数校验</th>
                <th>存在性校验</th>
                <th>一致性校验</th>
                <th>特性交叉校验</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>softmax_precision</td>
                <td>
                    <ul>
                        <li>data_type支持INT</li>
                        <li>Ascend 950非量化FLOAT16/BFLOAT16场景支持0（高精度）和1（混合精度）</li>
                        <li>Ascend 950 FP8量化场景仅支持1</li>
                    </ul>
                </td>
                <td>可选属性，默认值为1</td>
                <td>metadata接口与主算子须一致</td>
                <td>无</td>
            </tr>
        </tbody>
    </table>

#### Mask参数组

mask_mode参数解释见「mask_mode 枚举」。

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>mask_mode</td>
            <td>
                <ul>
                    <li>data_type支持INT</li>
                    <li>当前仅支持1</li>
                </ul>
            </td>
            <td>可选输入，默认值为1</td>
            <td rowspan="4">
                <ul>
                    <li>metadata接口与主算子的mask_mode须一致</li>
                    <li>当前win_left和win_right须为-1</li>
                </ul>
            </td>
            <td rowspan="4">
                <ul>
                    <li>mask_mode=1时使用算子内置causal mask</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>attn_mask</td>
            <td>
                <ul>
                    <li>当前须为None</li>
                </ul>
            </td>
            <td>无</td>
        </tr>
        <tr>
            <td>win_left</td>
            <td rowspan="2">
                <ul>
                    <li>data_type支持INT</li>
                    <li>当前仅支持-1</li>
                </ul>
            </td>
            <td rowspan="2">可选输入，默认值为-1</td>
        </tr>
        <tr>
            <td>win_right</td>
        </tr>
    </tbody>
</table>

#### SeqLengths参数组

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>seqused_q</td>
            <td rowspan="2">
                <ul>
                    <li>tensor_type支持int32</li>
                    <li>tensor_shape为(B,)</li>
                    <li>仅支持非负整数</li>
                    <li>seqused_q中的值需小于等于Q_S</li>
                    <li>seqused_kv中的值需小于等于KV_S</li>
                </ul>
            </td>
            <td rowspan="2">可选参数</td>
            <td rowspan="4">无</td>
            <td rowspan="4">无</td>
        </tr>
        <tr>
            <td>seqused_kv</td>
        </tr>
        <tr>
            <td>cu_seqlens_q</td>
            <td rowspan="2">
                <ul>
                    <li>tensor_type支持int64</li>
                    <li>tensor_shape为(B+1,)</li>
                    <li>值仅支持非负整数</li>
                    <li>其值应非递减（大于等于前一个值）排列</li>
                    <li>cu_seqlens_q：第一个元素为0且最后一个元素等于Q_T</li>
                    <li>cu_seqlens_kv：第一个元素为0且最后一个元素等于KV_T</li>
                </ul>
            </td>
            <td rowspan="2">
                <ul>
                    <li>当layout_q为TND时，cu_seqlens_q必须传入</li>
                    <li>当layout_q不为TND时，cu_seqlens_q不支持传入</li>
                    <li>当layout_kv为TND时，cu_seqlens_kv必须传入</li>
                    <li>当layout_kv不为TND时，cu_seqlens_kv不支持传入</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cu_seqlens_kv</td>
        </tr>
        <tr>
            <td>max_seqlen_q</td>
            <td rowspan="2">
                <ul>
                    <li>data_type支持INT</li>
                    <li>默认值为-1</li>
                </ul>
            </td>
            <td rowspan="2">可选参数</td>
            <td rowspan="2">
                <ul>
                    <li>值必须大于等于-1；传入时必须等于实际的最大序列长度，否则行为未定义</li>
                </ul>
            </td>
            <td rowspan="2">无</td>
        </tr>
        <tr>
            <td>max_seqlen_kv</td>
        </tr>
        </tbody>
</table>

#### Paged Attention参数组

当block_table不为空时，开启Paged Attention
<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>block_table</td>
            <td>
                <ul>
                    <li>tensor_type仅支持int32</li>
                    <li>tensor_shape为(B, max_num_blocks_per_seq)</li>
                    <li>值只能为正整数</li>
                </ul>
            </td>
            <td>可选参数</td>
            <td>无</td>
            <td>
                <ul>
                    <li>PagedAttention开启情况下，必须传入seqused_kv</li>
                    <li>PagedAttention开启情况下，block_table必须不为空</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

#### Quant参数组

quant_mode参数解释见「quant_mode 枚举」。

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>quant_mode</td>
            <td>
                <ul>
                    <li>data_type支持INT</li>
                    <li>当前支持0、5</li>
                </ul>
            </td>
            <td>可选输入，默认值为0</td>
            <td rowspan="5">
                <ul>
                    <li>metadata接口与主算子的quant_mode须一致</li>
                </ul>
            </td>
            <td rowspan="5">
                <ul>
                    <li>quant_mode=0时，q/k/v_dequant_scale须为None</li>
                    <li>quant_mode!=0时attention_out_dtype必须传入，当前量化场景输出dtype须为float16或bfloat16</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>q_dequant_scale</td>
            <td rowspan="3">
                <ul>
                    <li>非量化场景须为None</li>
                </ul>
            </td>
            <td rowspan="3">可选参数</td>
        </tr>
        <tr>
            <td>k_dequant_scale</td>
        </tr>
        <tr>
            <td>v_dequant_scale</td>
        </tr>
        <tr>
            <td>p_quant_scale</td>
            <td>
                <ul>
                    <li>当前须为None</li>
                </ul>
            </td>
            <td>无</td>
        </tr>
    </tbody>
</table>

#### SoftmaxLSE参数组

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>return_softmax_lse</td>
            <td>
                <ul>
                    <li>data_type仅支持BOOL</li>
                    <li>quant_mode=0支持False/True；FP8路径仅支持False</li>
                </ul>
            </td>
            <td>可选属性，默认值为False</td>
            <td rowspan="2">
                <ul>
                    <li>return_softmax_lse为False时，softmax_lse输出空Tensor</li>
                    <li>return_softmax_lse为True时，输出shape见<a href="#layout匹配关系表">layout匹配关系表</a></li>
                </ul>
            </td>
            <td>无</td>
        </tr>
        <tr>
            <td>softmax_lse</td>
            <td>
                <ul>
                    <li>data_type仅支持float32</li>
                </ul>
            </td>
            <td>无</td>
            <td>无</td>
        </tr>
    </tbody>
</table>

## 调用示例

- generic_block_sparse_attention_metadata + generic_block_sparse_attention联合调用示例（TND + PA_BBND）

    ```python
    import math
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    B, Q_N, KV_N, Q_S, KV_S, D = 1, 32, 8, 128, 256, 128
    block_x, block_y = 1, 128
    block_size = 128
    top_k = 16

    Q_T = B * Q_S
    num_blocks = math.ceil(KV_S / block_size)
    total_q_blocks = math.ceil(Q_S / block_x)

    q = torch.randn(Q_T, Q_N, D, dtype=torch.float16, device="npu")
    k = torch.randn(num_blocks, block_size, KV_N, D, dtype=torch.float16, device="npu")
    v = torch.randn(num_blocks, block_size, KV_N, D, dtype=torch.float16, device="npu")

    sparse_block_idx = torch.randint(
        0, num_blocks, (KV_N, total_q_blocks, top_k), dtype=torch.int32, device="npu"
    )
    sparse_block_count = torch.full((KV_N, total_q_blocks), top_k, dtype=torch.int32, device="npu")
    cu_seqlens_q = torch.tensor([0, Q_S], dtype=torch.int64, device="npu")
    seqused_kv = torch.tensor([KV_S], dtype=torch.int32, device="npu")
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="npu").view(1, -1)
    block_shape = [block_x, block_y]

    metadata = cann_ops_transformer.ops.generic_block_sparse_attention_metadata(
        sparse_block_idx,
        sparse_block_count,
        Q_N,
        KV_N,
        D,
        block_shape,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        max_seqlen_q=Q_S,
        max_seqlen_kv=KV_S,
        is_packed_gqa=True,
        layout_q="TND",
        layout_kv="PA_BBND",
        mask_mode=1,
        quant_mode=0,
        softmax_precision=1,
    )

    attention_out, softmax_lse = cann_ops_transformer.ops.generic_block_sparse_attention(
        q,
        k,
        v,
        sparse_block_idx,
        sparse_block_count,
        block_shape,
        metadata=metadata,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        block_table=block_table,
        is_packed_gqa=True,
        layout_q="TND",
        layout_kv="PA_BBND",
        softmax_scale=1.0 / (D ** 0.5),
        mask_mode=1,
        quant_mode=0,
        softmax_precision=1,
        return_softmax_lse=False,
    )
    torch_npu.npu.synchronize()
    assert attention_out.shape == q.shape
    assert attention_out.dtype == q.dtype
    ```
