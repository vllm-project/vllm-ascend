# MegaMoe

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                               |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

算子功能：该算子将`MoE（Mixture of Experts）`以及`FFN（Feed-Forward Network）`的完整计算流程融合为单个算子，实现`Dispatch + GroupMatmul1 + SwiGLUQuant + GroupMatmul2 + Combine`的端到端融合计算。

计算公式：

  MegaMoe是一个多阶段融合的MoE算子，在专家并行场景下依次完成Token路由分发与量化、分组矩阵乘法 + SwiGLU激活、二次分组矩阵乘法与跨Rank聚合、以及加权求和，完整计算流程可以分解为以下四个阶段。

  第一阶段对输入Token按专家分组收集后做MXFP8量化，生成各专家的量化输入与缩放因子：

  $$
  \hat{X}_e,\ S_{X,e} = \mathrm{Q}_{\text{MX}}\!\left(X[\mathcal{T}_e]\right), \quad e = 0, 1, \ldots, E_{\text{local}}-1
  $$

  说明：根据`topkIds`将Token按专家排序收集，$\mathcal{T}_e$ 为分配到专家 $e$ 的Token索引集合，$E_{local}$ 表示当前专家收到的最大Token数，每个专家数值可能不同，$X[\mathcal{T}_e]$ 为对应的子矩阵。$\mathrm{Q}_{\text{MX}}$ 表示MX逐组量化（group size = 32），对每组32个元素提取共享指数后量化为FP8目标类型（FLOAT8_E5M2或FLOAT8_E4M3FN），同时输出FLOAT8_E8M0缩放因子。量化后的数据将作为GMM1的输入。

  第二阶段对每个专家执行GMM1矩阵乘法（将 $W_1$ 沿列方向分为两半分别计算）、SwiGLU激活和MX量化：

  $$
  Z_e^{(x)} = \mathrm{DQ}_{\text{MX}}(\hat{X}_e, S_{X,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{1,e}^{(x)}, S_{1,e}^{(x)}), \quad Z_e^{(y)} = \mathrm{DQ}_{\text{MX}}(\hat{X}_e, S_{X,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{1,e}^{(y)}, S_{1,e}^{(y)})
  $$

  $$
  U_e = Z_e^{(x)} \odot \sigma\!\left(Z_e^{(x)}\right) \odot Z_e^{(y)}
  $$

  $$
  \hat{U}_e,\ S_{U,e} = \mathrm{Q}_{\text{MX}}(U_e)
  $$

  说明：将 $W_1$ 的前 $N/2$ 列 $W_{1,e}^{(x)}$ 和后 $N/2$ 列 $W_{1,e}^{(y)}$ 分别与MX反量化后的输入做矩阵乘法，得到Swish分支 $Z_e^{(x)}$ 和门控分支 $Z_e^{(y)}$。SwiGLU激活对两个分支做逐元素乘积 $x \cdot \sigma(x) \cdot y$，其中 $\sigma$ 为Sigmoid函数，将中间维度从 $N$ 减半为 $N/2$。随后对SwiGLU输出做MX量化，得到GMM2的量化输入 $\hat{U}_e$。

  第三阶段对每个专家执行GMM2矩阵乘法，并将结果按目标Rank分发：

  $$
  O_e = \mathrm{DQ}_{\text{MX}}(\hat{U}_e, S_{U,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{2,e}, S_{2,e})
  $$

  说明：将量化后的SwiGLU输出与第二组权重 $W_2$ 做MX反量化后的矩阵乘法，将 $N/2$ 维中间表示映射回 $H$ 维隐藏空间，得到每个专家的输出 $O_e$。计算完成后通过RDMA peermem将结果按目标Rank的专家偏移地址写入远端，实现跨Rank聚合。

  第四阶段对所有Token按路由权重加权求和，恢复为与输入相同形状的输出：

  $$
  Y[i] = \sum_{k=0}^{K-1} W[i,\, k] \cdot O[\pi(i,\, k)]
  $$

  说明：对每个Token $i$，根据排序后的路由索引 $\pi(i,k)$ 从聚合后的专家结果中取出对应行，按`topkWeights`中的权重逐元素加权累加，得到最终输出 $Y$。

  其中，$X$ 表示参数`x`，$W$ 表示参数`topkWeights`，$W_1$ 表示参数`weight1`，$W_2$ 表示参数`weight2`，$Y$ 表示参数`y`，$E_{\text{local}}$ 表示属性`moeExpertNum / epWorldSize`（每个Rank的专家数），$K$ 表示`topkIds`的第二维度（top-K值，取值6或8），$N$ 表示`hiddenDim（weight1.dim1）`。

  局部变量说明：
  - $\mathcal{T}_e$：被路由到专家 $e$ 的Token索引集合，由`topkIds`排序后确定。
  - $\hat{X}_e,\ S_{X,e}$：专家 $e$ 的量化输入及其MX缩放因子，第一阶段中间结果。
  - $W_{1,e}^{(x)}$、$W_{1,e}^{(y)}$：$W_1$ 对应专家 $e$ 的前 $N/2$ 列和后 $N/2$ 列子矩阵，由`weight1`按SwiGLU拆分推导。
  - $S_{1,e}^{(x)}$、$S_{1,e}^{(y)}$：$W_{1,e}^{(x)}$ 和 $W_{1,e}^{(y)}$ 对应的MX缩放因子，从`weightScales1`按维度截取。
  - $S_{2,e}$：$W_{2,e}$ 对应的MX缩放因子，来自参数`weightScales2`。
  - $Z_e^{(x)},\ Z_e^{(y)}$：GMM1的两路矩阵乘法输出（Swish分支和门控分支），中间结果。
  - $U_e$：SwiGLU激活输出，维度 $m_e \times N/2$，中间结果。
  - $\hat{U}_e,\ S_{U,e}$：量化后的SwiGLU输出及其MX缩放因子，中间结果。
  - $O_e$：GMM2的专家级输出，维度 $m_e \times H$，中间结果。
  - $\pi(i, k)$：Token $i$ 的第 $k$ 个top-k专家在展开排序后的行索引，由路由排序确定。
  - $\mathrm{Q}_{\text{MX}}(\cdot)$：MX逐组量化操作，block size = 32，输出FP8数据和E8M0缩放因子。
  - $\mathrm{DQ}_{\text{MX}}(\cdot)$：MX逐组反量化操作，在matmul内部隐式执行。



## 参数说明

<table style="undefined;table-layout: fixed; width: 1392px"> <colgroup>
 <col style="width: 120px">
 <col style="width: 120px">
 <col style="width: 160px">
 <col style="width: 150px">
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
   <td>context</td>
   <td>输入</td>
   <td>本卡通信域信息数据。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>x</td>
   <td>输入</td>
   <td>本卡发送的token数据。</td>
   <td>BF16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>topk_ids</td>
   <td>输入</td>
   <td>每个token的topK个专家索引。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>topk_weights</td>
   <td>输入</td>
   <td>每个token的topK个专家权重。</td>
   <td>FP32、BF16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight1</td>
   <td>输入</td>
   <td>GroupMatmul1计算的右矩阵，用于计算SwiGLU激活前的线性变换。</td>
   <td>FP8_E5M2、FP8_E4M3</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight2</td>
   <td>输入</td>
   <td>GroupMatmul2计算的右矩阵，用于SwiGLU激活后的线性变换。</td>
   <td>FP8_E5M2、FP8_E4M3</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight_scales1</td>
   <td>可选输入</td>
   <td>量化场景需要，GroupMatmul1右矩阵反量化参数。</td>
   <td>FP8_E8M0</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weight_cales2</td>
   <td>可选输入</td>
   <td>量化场景需要，GroupMatmul2右矩阵反量化参数。</td>
   <td>FP8_E8M0</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>x_active_mask</td>
   <td>可选输入</td>
   <td>预留参数，暂不支持。</td>
   <td>INT8</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>scales</td>
   <td>可选输入</td>
   <td>预留参数，暂不支持。</td>
   <td>FP32、FP8_E8M0</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>moe_expert_num</td>
   <td>属性</td>
   <td>MoE模型的总专家数量。</td>
   <td>INT64</td>
   <td></td>
  </tr> 
  <tr>
   <td>ep_world_size</td>
   <td>属性</td>
   <td>专家并行通信域大小。</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>ccl_buffer_size</td>
   <td>属性</td>
   <td>CCL通信缓冲区大小。</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>max_recv_token_num</td>
   <td>可选属性</td>
   <td>每个Rank最大可接收Token数。默认值为0。该值为0时，会按最大值bs*ep_world_size*min(topK,expertPerRank)预留内存大小；不为0时，将按照输入值大小预留内存，该场景下需用户保证填入值大于等于每个rank最大可接收的Token数。</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>dispatch_quant_mode</td>
   <td>可选属性</td>
   <td>dispatch通信时量化模式。目前仅支持4（MXFP模式）。默认值为0。</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>dispatch_quant_out_type</td>
   <td>可选属性</td>
   <td>dispatch量化后输出的数据类型。支持23（FP8_E5M2）、24（FP8_E4M3）。</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>combine_quant_mode</td>
   <td>可选属性</td>
   <td>预留参数，暂不支持。默认值为0。</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>comm_alg</td>
   <td>可选属性</td>
   <td>预留参数，暂不支持。默认值为""。</td>
   <td>STRING</td>
   <td></td>
  </tr>
  <tr>
   <td>global_bs</td>
   <td>可选属性</td>
   <td>全局batch size，多卡场景下的总Token数，默认值为0表示使用单卡BS。默认值按最大值maxBs*ep_world_size计算。</td>
   <td>INT64</td>
   <td></td>
  </tr>
  <tr>
   <td>y</td>
   <td>输出</td>
   <td>计算输出结果，与输入x shape相同。</td>
   <td>BF16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expert_token_nums</td>
   <td>输出</td>
   <td>每个专家收到的token数量。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>


## 约束说明

- **参数一致性约束**：
    - 调用算子过程中使用的`ep_world_size`、`global_bs`、`HCCL_BUFFSIZE`等参数取值，所有卡需保持一致，网络中不同层中也需保持一致。

- **通信域使用约束**：
    - 仅支持`EP`域，无`TP`域，不支持`groupTp`、`tpWorldSize`、`tpRankId`属性。
    - 所有卡的`moe_expert_num`、`ep_world_size`、`ccl_buffer_size`、`max_recv_token_num`、`dispatch_quant_mode`、`dispatch_quant_out_type`、`global_bs`参数取值需保持一致。

- **参数约束**：
  - BS（x.dim0）范围 [1, 512]。
  - H（x.dim1）仅支持4096、5120、7168。
  - topK（topkIds.dim1）仅支持6或8。
  - expertPerRank（weight1.dim0）范围 [1, 16]。
  - hiddenDim（weight1.dim1）仅支持1024、2048、3072、4096、7168。
  - epWorldSize范围 [2, 768]。
  - moeExpertNum范围 [epWorldSize, 1024]，且moeExpertNum % epWorldSize == 0。
  - maxRecvTokenNum范围 [0, BS × epWorldSize × min(topK, expertPerRank)]。
  - dispatchQuantOutType仅支持23（FLOAT8_E5M2）或24（FLOAT8_E4M3FN）。
  - globalBs为0或满足BS × epWorldSize <= globalBs且globalBs % epWorldSize == 0。
  - 当前版本仅支持MXFP量化模式（dispatchQuantMode = 4），dispatch阶段使用MX逐组量化（group size = 32），量化缩放因子类型为FLOAT8_E8M0。
  - combineQuantMode必须为0，commAlg必须为空字符串""。
  - y的数据类型与x相同。
  - weight1的dim1（hiddenDim）必须等于weight2的dim2的二倍，这是因为SwiGLU激活需要将中间维度从hiddenDim减半为hiddenDim/2。
  - expertPerRank = moeExpertNum / epWorldSize，必须为整数且在 [1, 16] 范围内。

- **MXFP量化场景约束**：
    - weight1 shape为(expertPerRank, hiddenDim, H)，weight2 shape为(expertPerRank, H, hiddenDim/2)。
    - weightScales1 shape为(expertPerRank, hiddenDim, CeilDiv(H, 64), 2)，其中CeilDiv(H, 64) = (H + 63) / 64。
    - weightScales2 shape为(expertPerRank, H, CeilDiv(hiddenDim/2, 64), 2)，其中CeilDiv(hiddenDim/2, 64) = (hiddenDim/2 + 63) / 64。
    - weightScales1的dim3和weightScales2的dim3必须等于2。
    - MXFP场景下，dispatchQuantOutType=23时weight1和weight2必须为FLOAT8_E5M2，dispatchQuantOutType=24时必须为FLOAT8_E4M3FN。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| PyTorch接口调用 | [deepep.py](../../torch_extension/cann_ops_transformer/ops/deep_ep.py) | 通过[mega_moe](../../torch_extension/cann_ops_transformer/docs/zh/mega_moe.md)PyTorch接口方式调用mega_moe算子。 |
