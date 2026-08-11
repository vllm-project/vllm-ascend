# KvQuantSparseFlashAttention

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列加速卡产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- API功能：`kv_quant_sparse_flash_attention`在`sparse_flash_attention`的基础上支持了[Per-Token-Head-Tile-128量化]输入。随着大模型上下文长度的增加，Sparse Attention的重要性与日俱增，这一技术通过“只计算关键部分”大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加，进而影响整体性能。

- 计算公式：

    $$
    Attention=\text{softmax}(\frac{Q @ \text{Dequant}({\tilde{K}^{INT8}},{Scale_K})^T}{\sqrt{d_k}})@\text{Dequant}(\tilde{V}^{INT8},{Scale_V}),
    $$

    其中$\tilde{K},\tilde{V}$为基于某种选择算法（如`LightningIndexer`）得到的重要性较高的Key和Value，一般具有稀疏或分块稀疏的特征，$d_k$为$Q,\tilde{K}$每一个头的维度，$\text{Dequant}(\cdot,\cdot)$为反量化函数。
本次公布的`kv_quant_sparse_flash_attention`是面向Sparse Attention的全新算子，针对离散访存进行了指令缩减及搬运聚合的细致优化。

## 参数说明

> **说明：**<br>
> 参数维度含义：B表示Batch Size、Q_S和KV_S分别表示query和key/value的Sequence Length、Q_N和KV_N分别表示query和key/value的Head Num、Q_D和KV_D分别表示query和key/value的Head Dim、Q_T和KV_T分别表示query和key/value的Total Tokens、sparse_size表示一次离散选取的block数、block_num和block_size分别表示PageAttention场景下的block总数和每个block的token数。

<table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
      <tbody>
      <tr>
          <td>query</td>
          <td>输入</td>
          <td>attention结构的Q输入，不支持非连续。query由相同数据类型的q_nope和q_rope按D维度拼接得到。layout_query为"BSND"时shape为[B, Q_S, Q_N, Q_D]。layout_query为"TND"时shape为[Q_T, Q_N, Q_D]。其中Q_D值仅支持576，即q_nope+q_rope=512+64；Q_N值支持1/2/4/8/16/32/48/64/128。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>key</td>
          <td>输入</td>
          <td>attention结构的K输入，不支持非连续。k_nope、query相同数据类型的k_rope和float32的量化参数按D维度拼接得到。layout_kv为"BSND"时shape为[B, KV_S, KV_N, KV_D]。layout_kv为"TND"时shape为[KV_T, KV_N, KV_D]。layout_kv为"PA_BSND"时shape为[block_num, block_size, KV_N, KV_D]，其中block_num为PageAttention时block总数，block_size为一个block的token数，block_size取值为16的整数倍，最大支持到1024。KV_N仅支持1；KV_D值仅支持656，即nope+rope*2+dequant_scale*4=512+64*2+4*4。</td>
          <td>FLOAT8_E4M3、INT8、HIFLOAT8</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>value</td>
          <td>输入</td>
          <td>attention结构的V输入，不支持非连续。</td>
          <td>FLOAT8_E4M3、INT8、HIFLOAT8</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>sparse_indices</td>
          <td>输入</td>
          <td>代表离散取kvCache的索引，不支持非连续。layout_query为"BSND"时shape为[B, Q_S, KV_N, sparse_size]。layout_query为"TND"时shape为[Q_T, KV_N, sparse_size]。其中sparse_size为一次离散选取的block数，需要保证每行有效值均在前半部分，无效值均在后半部分，且需要满足sparse_size大于0。当key和value的数据类型为hifloat8时，sparse_size仅支持2048。</td>
          <td>INT32</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>scale_value</td>
          <td>属性</td>
          <td>公式中d<sub>k</sub>开根号的倒数，代表缩放系数，作为query和key矩阵乘后Muls的scalar值。</td>
          <td>FLOAT</td>
          <td>-</td>
      </tr>
      <tr>
          <td>key_quant_mode</td>
          <td>属性</td>
          <td>代表key的量化模式，仅支持传入2，代表per_tile量化模式。</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>value_quant_mode</td>
          <td>属性</td>
          <td>代表value的量化模式，仅支持传入2，代表per_tile量化模式。</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>key_dequant_scale</td>
          <td>输入</td>
          <td>预留参数。</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>value_dequant_scale</td>
          <td>输入</td>
          <td>预留参数。</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>block_table</td>
          <td>输入</td>
          <td>表示PageAttention中kvCache存储使用的block映射表。shape为[B, KV_S_max/block_size]，其中第一维长度为B，第二维长度不小于所有batch中最大的KV_S对应的block数量，即KV_S_max / block_size向上取整。</td>
          <td>INT32</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>actual_seq_lengths_query</td>
          <td>输入</td>
          <td>表示不同Batch中query的有效token数。如果不指定seqlen可传入None，表示和query shape的Q_S长度相同。shape为[B,]。每个Batch的有效token数不超过query中的Q_S大小且不小于0。当layout_query为"TND"时，该入参必须传入，且以该入参元素的数量作为B值，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</td>
          <td>INT32</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>actual_seq_lengths_kv</td>
          <td>输入</td>
          <td>表示不同Batch中key和value的有效token数。如果不指定None，表示和key的shape的KV_S长度相同。shape为[B,]。每个Batch的有效token数不超过key/value中的KV_S大小且不小于0。当layout_kv为"TND"或"PA_BSND"时，该入参必须传入，layout_kv为"TND"时，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</td>
          <td>INT32</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>sparse_block_size</td>
          <td>属性</td>
          <td>代表sparse阶段的block大小。sparse_block_size为1时，为Token-wise稀疏化场景；sparse_block_size大于1且小于等于128时，为Block-wise稀疏化场景，块内token共享相同的稀疏化决策。</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>layout_query</td>
          <td>属性</td>
          <td>用于标识输入query的数据排布格式，默认值"BSND"，支持传入BSND和TND。</td>
          <td>STRING</td>
          <td>-</td>
      </tr>
      <tr>
          <td>layout_kv</td>
          <td>属性</td>
          <td>用于标识输入key的数据排布格式，默认值"BSND"，支持传入BSND、TND和PA_BSND，PA_BSND在开启PageAttention时使用。</td>
          <td>STRING</td>
          <td>-</td>
      </tr>
      <tr>
          <td>sparse_mode</td>
          <td>属性</td>
          <td>表示sparse的模式。sparse_mode为0时，代表全部计算。sparse_mode为3时，代表rightDownCausal模式的mask，对应以右下顶点往左上为划分线的下三角场景。</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>pre_tokens</td>
          <td>属性</td>
          <td>用于稀疏计算，表示attention需要和前几个Token计算关联，仅支持2^63-1。</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>next_tokens</td>
          <td>属性</td>
          <td>用于稀疏计算，表示attention需要和后几个Token计算关联，仅支持2^63-1。</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>attention_mode</td>
          <td>属性</td>
          <td>表示attention的模式，仅支持传入2，表示MLA-absorb模式，即QK的D包含rope和nope两部分，且KV是同一份。</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>quant_scale_repo_mode</td>
          <td>属性</td>
          <td>表示量化参数的存放模式，仅支持传入1，表示combine模式，即量化参数和数据混合存放。</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>tile_size</td>
          <td>属性</td>
          <td>表示per_tile时每个参数对应的数据块大小，仅在per_tile时有效，仅支持128。</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>rope_head_dim</td>
          <td>属性</td>
          <td>表示MLA架构下的rope_head_dim大小，仅在attention_mode为2时有效，仅支持64。</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>output</td>
          <td>输出</td>
          <td>代表公式中的输出Attention。输出shape与入参query的shape保持一致，layout_query为"BSND"时shape为[B, Q_S, Q_N, Q_out_D]，layout_query为"TND"时shape为[Q_T, Q_N, Q_out_D]，其中Q_out_D = Q_D - rope_head_dim。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
      </tr>
      </tbody>
  </table>

## 约束说明

- 该接口支持图模式。
- 参数query shape中：<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：Q_N不支持48。
- 参数key、value数据类型要求：
    - <term>Ascend 950PR/Ascend 950DT</term>：仅支持float8_e4m3、int8、hifloat8数据类型。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：仅支持int8数据类型。
- 参数sparse\_block\_size：
    - <term>Ascend 950PR/Ascend 950DT</term>：只支持sparse\_block\_size为1。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持[1,16]，且要求是2的幂次方，在PageAttention场景下要求sparse\_block\_size整除block\_size
- 非PageAttention场景layout\_query和layout\_kv取值需要保持一致。
