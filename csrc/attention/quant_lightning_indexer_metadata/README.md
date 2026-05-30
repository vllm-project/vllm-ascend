# QuantLightningIndexerMetadata

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>   | √  |
|<term>Atlas A3 推理系列产品</term>   | √  |
|<term>Atlas A2 推理系列产品</term>   | √  |

## 功能说明

- API功能：QuantLightningIndexerMetadata是QuantLightningIndexer的前置算子，通过AICPU为QuantLightningIndexer算子生成分核结果，包括每个核需要处理的数据的起始点、结束点等内容，随后，QuantLightningIndexer根据该分核结果进行实际计算。

- 主要计算过程为：
    1. 获取每个`batch`的基本块大小，并计算负载。
    2. 计算所有`batch`的总负载和总的基本块个数。
    3. 为每个核分配负载，并记录分核结果，分核结果包括每个核需要处理的数据的起始点、结束点等内容。

## 参数说明

>- 参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
>- 使用S1和S2分别表示query和key的输入样本序列长度，N1和N2分别表示query和key对应的多头数，k表示最后选取的索引个数。

 <table style="undefined;table-layout: fixed; width: 1000px">
       <colgroup>
       <col style="width: 100px">
       <col style="width: 120px">
       <col style="width: 500px">
       <col style="width: 80px">
       <col style="width: 80px">
       </colgroup>
       <thead>
         <tr>
           <th>参数名</th>
           <th>输入/输出/属性</th>
           <th>描述</th>
           <th>数据类型</th>
           <th>数据格式</th>
         </tr>
       </thead>
       <tbody>
         <tr>
           <td>num_heads_q</td>
           <td>属性</td>
           <td>Q的多头数。</td>
           <td>INT32</td>
           <td>-</td>
         </tr>
         <tr>
           <td>num_heads_k</td>
           <td>属性</td>
           <td>K的多头数。</td>
           <td>INT32</td>
           <td>-</td>
         </tr>
         <tr>
           <td>head_dim</td>
           <td>属性</td>
           <td>注意力头的维度。</td>
           <td>INT32</td>
           <td>-</td>
         </tr>
         <tr>
           <td>query_quant_mode</td>
           <td>属性</td>
           <td>用于标识query的量化模式，当前支持Per-Token-Head量化模式，当前仅支持传入0</td>
           <td>INT32</td>
           <td>-</td>
         </tr>
         <tr>
           <td>key_quant_mode</td>
           <td>属性</td>
           <td>用于标识输入key的量化模式，当前支持Per-Token-Head量化模式，当前仅支持传入0。</td>
           <td>INT32</td>
           <td>-</td>
         </tr>
         <tr>
           <td>actual_seq_lengths_query</td>
           <td>可选输入</td>
           <td>表示不同Batch中`query`的有效token数，数据类型支持`int32`。如果不指定seqlen可传入None，表示和`query`的shape的S长度相同。该入参中每个Batch的有效token数不超过`query`中的维度S大小且不小于0。支持长度为B的一维tensor。<br>当`layout_query`为TND时，该入参必须传入，且以该入参元素的数量作为B值，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。不能出现负值。特殊说明：actual\_seq\_lengths\_query和actual\_seq\_lengths\_key至少传入一个。</td>
           <td>INT32</td>
           <td>-</td>
         </tr>
         <tr>
           <td>actual_seq_lengths_key</td>
           <td>可选输入</td>
           <td>表示不同Batch中压缩前原始`key`的有效token数，数据类型支持`int32`。如果不指定seqlen可传入None，表示和key的shape的S长度相同。该参数中每个Batch的原始有效token数除以压缩率后不超过`key`中的维度S大小且不小于0，支持长度为B的一维tensor。<br>当`layout_kv`为TND或PA_BSND时，该入参必须传入，`layout_kv`为TND，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。特殊说明：actual\_seq\_lengths\_query和actual\_seq\_lengths\_key至少传入一个。</td>
           <td>INT32</td>
           <td>-</td>
         </tr>
         <tr>
           <td>batch_size</td>
           <td>可选属性</td>
           <td>输入样本批量大小。</td>
           <td>INT32</td>
           <td>-</td>
         </tr>
         <tr>
           <td>max_seqlen_q</td>
           <td>可选属性</td>
           <td>当layout_query为BSND时，表示每个Batch中的q的有效token数。</td>
           <td>INT32</td>
           <td>-</td>
         </tr>
         <tr>
           <td>max_seqlen_k</td>
           <td>可选属性</td>
           <td>当layout_kv为BSND时，表示每个Batch中的k的有效token数。</td>
           <td>INT32</td>
           <td>-</td>
         </tr>
         <tr>
           <td>cmp_ratio</td>
           <td>可选属性</td>
           <td>用于稀疏计算，表示key的压缩倍数。Atlas A3 推理系列产品支持1/2/4/8/16/32/64/128，Ascend 950PR/Ascend 950DT支持1/4/128。数据类型支持int32，默认值1。</td>
           <td>INT32</td>
           <td>-</td>
         </tr>
         <tr>
           <td>layout_query</td>
           <td>可选属性</td>
           <td>用于标识query的数据排布格式，当前支持BSND、TND，默认值"BSND"。</td>
           <td>STRING</td>
         </tr>
         <tr>
           <td>layout_key</td>
           <td>可选属性</td>
           <td>用于标识key的数据排布格式，当前仅支持PA_BSND。</td>
           <td>STRING</td>
           <td>-</td>
         </tr>
         <tr>
           <td>sparse_count</td>
           <td>可选属性</td>
           <td>代表topK阶段需要保留的block数量，Atlas A3推理系列产品支持[1, 2048]，Ascend 950PR/Ascend 950DT支持512。</td>
           <td>INT32</td>
           <td>-</td>
         </tr>
         <tr>
           <td>sparse_mode</td>
           <td>可选属性</td>
           <td>表示sparse的模式，支持0/3，数据类型支持int32。为0时，代表defaultMask模式。为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景.</td>
           <td>INT32</td>
           <td>-</td>
         </tr>
         <tr>
           <td>pre_token</td>
           <td>可选属性</td>
           <td>用于稀疏计算，表示attention需要和前几个Token计算关联,仅支持默认值2^63-1。</td>
           <td>INT64</td>
           <td>-</td>
         </tr>
         <tr>
           <td>next_token</td>
           <td>可选属性</td>
           <td>用于稀疏计算，表示attention需要和前几个Token计算关联，仅支持默认值2^63-1。</td>
           <td>INT64</td>
           <td>-</td>
         </tr>
         <tr>
           <td>device</td>
           <td>可选属性</td>
           <td>npu的ID。</td>
           <td>STRING</td>
           <td>-</td>
         </tr>
         <tr>
           <td>metadata</td>
           <td>输出</td>
           <td>QuantLightningIndexerMetadata算子传入的分核信息，包括每个Cube核上FlashAttention计算任务的Batch、Head以及Q和K分块的索引，以及每个Vector核上FlashDecode的规约任务索引。数据类型支持`int32`，shape大小为[1024]</td>
           <td>INT32</td>
           <td>-</td>
         </tr>
       </tbody>
     </table>

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持aclgraph模式。
