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
          <td>attention结构的Q输入，不支持非连续。query由相同数据类型的q_nope和q_rope按D维度拼接得到。layout_query为"BSND"时shape为[B, Q_S, Q_N, Q_D]。layout_query为"TND"时shape为[Q_T, Q_N, Q_D]。其中Q_D = 512 + rope_head_dim。A2/A3支持Q_D为512（rope_head_dim=0）或576（rope_head_dim=64）；A5 INT8和FLOAT8_E4M3FN场景同样支持这两种Q_D，HiFloat8场景仍仅支持576。rope_head_dim=0时query仅包含q_nope；Q_N值支持1/2/4/8/16/32/48/64/128。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>key</td>
          <td>输入</td>
          <td>attention结构的K输入，不支持非连续。k_nope、query相同数据类型的k_rope和float32的量化参数按D维度拼接得到。layout_kv为"BSND"时shape为[B, KV_S, KV_N, KV_D]。layout_kv为"TND"时shape为[KV_T, KV_N, KV_D]。layout_kv为"PA_BSND"时shape为[block_num, block_size, KV_N, KV_D]，其中block_num为PageAttention时block总数，block_size为一个block的token数，block_size取值为16的整数倍，最大支持到1024。KV_N仅支持1；KV_D = 512 + rope_head_dim*2 + 4*4，表示每行拼接数据的字节数。A2/A3 INT8 C8支持528（rope_head_dim=0）或656（rope_head_dim=64）；A5 INT8和FLOAT8_E4M3FN场景同样支持528或656，HiFloat8场景仍仅支持656。4个FLOAT32 scale组成的区域起始偏移分别为512或640字节，偏移从0开始。</td>
          <td>FLOAT8_E4M3FN、INT8、HIFLOAT8</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>value</td>
          <td>输入</td>
          <td>attention结构的V输入，不支持非连续。A2/A3 C8 MLA场景下，有效V为key反量化后的512维NoPE部分。算子调用时key和value可复用同一份packed KV缓存，物理最后一维为528（rope_head_dim=0）或656（rope_head_dim=64）；原独立算子测试也支持传入最后一维为512的NoPE value张量。输出最后一维始终为512。</td>
          <td>FLOAT8_E4M3FN、INT8、HIFLOAT8</td>
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
          <td>公式中d<sub>k</sub>开根号的倒数，代表缩放系数，作为query和key矩阵乘后Muls的scalar值。rope_head_dim变化时继续使用调用方传入的scale_value。</td>
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
          <td>表示attention的模式，仅支持传入2，表示MLA-absorb模式，即QK的D由512维NoPE和可选的RoPE部分组成，且KV是同一份。</td>
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
          <td>表示MLA架构下的RoPE维度，仅在attention_mode为2时有效。A2/A3 INT8 C8支持0或64：0表示省略输入中的RoPE分支；A5 INT8和FLOAT8_E4M3FN场景同样支持0或64，HiFloat8场景仍仅支持64。默认值保持64。</td>
          <td>INT64</td>
          <td>-</td>
      </tr>
      <tr>
          <td>return_softmax_lse</td>
          <td>属性</td>
          <td>默认False。A2/A3为True时返回softmax_max和softmax_sum；有效query行的LSE可由softmax_max + log(softmax_sum)计算。rope_head_dim为0或64时均保留此行为。A5当前仅写attention输出，应使用False。</td>
          <td>BOOL</td>
          <td>-</td>
      </tr>
      <tr>
          <td>output</td>
          <td>输出</td>
          <td>代表公式中的输出Attention。输出的token/head维度与query一致，最后一维为512。layout_query为"BSND"时shape为[B, Q_S, Q_N, Q_out_D]，layout_query为"TND"时shape为[Q_T, Q_N, Q_out_D]，其中Q_out_D = Q_D - rope_head_dim。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
      </tr>
      <tr>
          <td>softmax_max / softmax_sum</td>
          <td>输出</td>
          <td>return_softmax_lse为False时均为空张量；A2/A3为True时，layout_query为"BSND"的shape为[B, KV_N, Q_S, Q_N/KV_N]，为"TND"的shape为[KV_N, Q_T, Q_N/KV_N]。</td>
          <td>FLOAT32</td>
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

## A2/A3 C8 NoPE（rope_head_dim=0）

本节描述A2/A3的INT8 C8路径；A5 INT8/FLOAT8_E4M3FN扩展见下方独立章节。两种输入合同如下：

| rope_head_dim | Q最后一维（元素） | packed Key每行（字节） | scale起始偏移（字节，从0开始） | 有效V / output最后一维 |
| --- | --- | --- | --- | --- |
| 0 | 512 | 528 = 512 INT8 NoPE + 4 FLOAT32 scale | 512 | 512 |
| 64 | 576 | 656 = 512 INT8 NoPE + 64 FP16/BF16 RoPE + 4 FLOAT32 scale | 640 | 512 |

表中的V维度指有效计算数据。算子调用时Key/Value可复用同一份528或656字节的packed缓存。

rope_head_dim=0表示输入中没有RoPE分支，不需要调用方补齐64维RoPE。
内核跳过RoPE输入读取，在内部原有计算区域补零，保留原计算分块和缓冲区大小。
这使紧凑输入与相同NoPE、scale、稀疏索引及页表下的显式零RoPE输入保持功能一致；
调用方传入的scale_value保持不变。输出仍为512维，return_softmax_lse和图捕获/重放接口保持不变。

### 测试入口

在仓库根目录、已编译并安装自定义算子的A2/A3环境中运行：

```bash
python -m pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_kv_quant_sparse_flash_attention.py
python -m pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_kv_quant_sparse_flash_attention_rope0_a2a3.py
```

原测试继续使用原有随机golden和精度阈值。新增[rope0测试](../../../tests/e2e/nightly/single_node/ops/singlecard_ops/test_kv_quant_sparse_flash_attention_rope0_a2a3.py)
共53项，仅在A2/A3硬件配置上运行，覆盖FP16/BF16、rope0/64、PA与TND、batch与尾块、
LSE开关、图捕获/修改KV后的重放，以及非法RoPE维度和输入shape。
随机紧凑输入与显式补零输入做逐位对照；均匀attention用例以独立计算的选中V均值验证精度，
并校验有效query行的softmax_max、softmax_sum及LSE。
两项Python接入测试调用真实RMSNorm/INT8量化、cache写入和attention路径，
验证空RoPE、FLOAT32 scale字节、slot=-1不写cache以及独立均值精度。

### Python SFA接入

A2/A3的C8 NoPE接入复用现有SFA接口：量化并打包INT8 NoPE和FLOAT32 scale，
跳过空RoPE旋转，再调用自定义QSFA算子。
缓存规格继续读取原有模型全局配置；本次不修改V1/V2的缓存分配逻辑。
模型配置与当前层的RoPE维度均为0时，packed缓存为528字节。
全局RoPE维度为64而当前层为0的混合配置，不在本次模型接入范围内。
浮点NoPE继续使用原有计算路径。本次不扩展C8 NoPE的上下文并行模型接入。

本次不新增A5专属模型入口。模型只有在选择SFA且
`enable_sparse_sfa_c8=True`时才会进入C8量化路径，因此不需要额外的设备类型拦截。

## Ascend 950 INT8/FP8 C8 NoPE（rope_head_dim=0）

A5复用现有C8接口和NoPE512布局，key/value数据类型支持INT8和FLOAT8_E4M3FN。
每128个NoPE元素共享一个FLOAT32 scale，每行包含4个scale。

| rope_head_dim | Q最后一维（元素） | packed Key每行（字节） | scale起始偏移（字节，从0开始） | output最后一维 |
| --- | --- | --- | --- | --- |
| 0 | 512 | 528 = 512 INT8/FP8 NoPE + 4 FLOAT32 scale | 512 | 512 |
| 64 | 576 | 656 = 512 INT8/FP8 NoPE + 64 FP16/BF16 RoPE + 4 FLOAT32 scale | 640 | 512 |

使用 `attention_mode=2`、`key_quant_mode=2`、`value_quant_mode=2`、
`quant_scale_repo_mode=1`、`tile_size=128` 和 `sparse_block_size=1`。
RoPE0场景下调用方无需提供补零的RoPE数据，`scale_value`保持模型原值。
A5当前只写attention输出，应使用 `return_softmax_lse=False`。
HiFloat8继续仅支持原有RoPE64路径。

内核按实际输入维度读取GM数据和scale，保留原有672字节UB行跨距，
并在缓冲区复用时清零缺失的RoPE区域。Q输入准备会先初始化内部NZ缓冲区，
再拷贝512个有效列。576维QK计算、512维value/output计算和原有softmax路径保持不变。

### 测试入口

在仓库根目录、已编译并安装A5自定义OPP和native extension的环境中运行：

```bash
python -m pytest --noconftest -o addopts= -v tests/e2e/nightly/single_node/ops/singlecard_ops/test_kv_quant_sparse_flash_attention_rope0_a5.py
```

测试覆盖INT8和FLOAT8_E4M3FN的紧凑RoPE0与显式补零RoPE64等价性、
独立精度参考、原有非零RoPE回归、多种layout和head数、修改KV后的图重放以及非法输入校验。
A5整模验证和性能优化不在本次算子改动范围内。
