# LightningIndexer

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：`lightning_indexer`基于一系列操作得到每一个token对应的Top-$k$个位置。

- 计算公式：

     $$
     Indices=\text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right]\right\}
     $$

     对于某个token对应的Index Query $Q_{index}\in\R^{g\times d}$，给定上下文Index Key $K_{index}\in\R^{S_{k}\times d},W\in\R^{g\times 1}$，其中$g$为GQA对应的group size，$d$为每一个头的维度，$S_{k}$是上下文的长度。

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 480px">
  <col style="width: 200px">
  <col style="width: 100px">
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
      <td>
          <ul>
                <li>公式中的输入Q。</li>
                <li>不支持空tensor和非连续。</li>
                <li>layout_query为BSND时，shape为(B,S1,N1,D)；layout_query为TND时，shape为(T1,N1,D)。</li>
          </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>
          <ul>
                <li>公式中的输入K。</li>
                <li>不支持空tensor和非连续。</li>
                <li>layout_key为PA_BSND时，shape为(block_num, block_size, N2, D)，其中block_num为PageAttention时block总数、block_size为一个block的token数；layout_key为BSND时，shape为(B, S2, N2, D)；layout_key为TND时，shape为(T2, N2, D)。</li>
          </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>输入</td>
      <td>
          <ul>
                <li>公式中的输入W。</li>
                <li>不支持空tensor和非连续。</li>
                <li>layout_query为BSND时，shape为(B,S1,N1)；layout_query为TND时，shape为(T1,N1)。</li>
          </ul>
      </td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actual_seq_lengths_query</td>
      <td>输入</td>
      <td>
          <ul>
                <li>每个Batch中Query的有效token数。</li>
                <li>不支持空tensor和非连续。</li>
                <li>可传入None表示与query的S长度相同。</li>
                <li>支持长度为B的一维tensor，且每个Batch的有效token数不超过query中的维度S大小且不小于0。layout_query为TND时该入参必须传入，并以元素数量作为B值。</li>
                <li>每个元素表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actual_seq_lengths_key</td>
      <td>输入</td>
      <td>
          <ul>
                <li>每个Batch中Key的有效token数。</li>
                <li>不支持空tensor和非连续。</li>
                <li>可传入None表示与key的S长度相同。</li>
                <li>支持长度为B的一维tensor，且每个Batch的有效token数不超过key/value中的维度S大小且不小于0。</li>
                <li>layout_key为TND或PA_BSND时该入参必须传入；其中layout_key为TND时，每个元素表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>block_table</td>
      <td>输入</td>
      <td>
          <ul>
                <li>表示PageAttention中KV存储使用的block映射表。</li>
                <li>不支持空tensor和非连续。</li>
                <li>PageAttention场景下，block_table必须为二维，第一维长度需要等于B，第二维长度不能小于maxBlockNumPerSeq（每个batch中最大actual_seq_lengths_key对应的block数量）。</li>
                <li>shape支持(B,S2/block_size)。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>layout_query</td>
      <td>属性</td>
      <td>
          <ul>
                <li>用于标识输入Query的数据排布格式。</li>
                <li>默认值为"BSND"，当前支持BSND、TND。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_key</td>
      <td>属性</td>
      <td>
          <ul>
                <li>用于标识输入Key的数据排布格式。</li>
                <li>默认值为"BSND"，当前支持PA_BSND、BSND、TND。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_count</td>
      <td>属性</td>
      <td>
          <ul>
                <li>topK阶段需要保留的block数量。</li>
                <li>支持[1, 2048]，以及3072、4096、5120、6144、7168、8192。</li>
                <li>默认值为2048。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_mode</td>
      <td>属性</td>
      <td>
          <ul>
                <li>表示sparse的模式。</li>
                <li>sparse_mode为0时代表defaultMask模式。</li>
                <li>sparse_mode为3时代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</li>
                <li>默认值为3。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pre_tokens</td>
      <td>属性</td>
      <td>用于稀疏计算，表示attention需要和前几个Token计算关联，仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>next_tokens</td>
      <td>属性</td>
      <td>用于稀疏计算，表示attention需要和后几个Token计算关联，仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>return_values</td>
      <td>属性</td>
      <td>
          <ul>
                <li>表示是否输出sparseValuesOut。</li>
                <li>True表示输出，False表示不输出，默认值为False。仅在训练且layout_key不为PA_BSND场景支持。</li>
          </ul>
      </td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_indices</td>
      <td>输出</td>
      <td>
          <ul>
                <li>公式中的Indices输出。</li>
                <li>不支持空tensor和非连续。</li>
                <li>layout_query为BSND时输出shape为[B, S1, N2, sparseCount]；layout_query为TND时输出shape为[T1, N2, sparseCount]。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_values</td>
      <td>输出</td>
      <td>
          <ul>
                <li>公式中的Indices输出对应的value值。</li>
                <li>不支持空tensor和非连续。</li>
                <li>shape与sparseIndicesOut保持一致。</li>
          </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- 该接口支持图模式。
- 参数key中的N2支持1。
- headdim支持128。
- block_size取值为16的倍数，最大支持1024。
- 参数query、key的数据类型应保持一致。
- 参数weights不为`float32`时，参数query、key、weights的数据类型应保持一致。
- Ascend 950PR/Ascend 950DT：
    - query N1仅支持8、16、24、32、64。
    - 参数weights不支持`float32`类型。
- A3 训练系列产品/Atlas A3 推理系列产品：
    - query N1支持小于等于64。

## 调用示例

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">调用方式</th>
    <th class="tg-0pky">样例代码</th>
    <th class="tg-0pky">说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">aclnn接口</td>
    <td class="tg-0pky">
    <a href="./examples//test_aclnn_lightning_indexer.cpp">test_aclnn_lightning_indexer
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    通过
    <a href="./docs/aclnnLightningIndexer.md">aclnnLightningIndexer
    </a>
    接口方式调用算子
    </td>
  </tr>
</tbody></table>
