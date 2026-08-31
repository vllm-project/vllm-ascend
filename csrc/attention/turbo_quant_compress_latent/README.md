# TurboQuantCompressLatent

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

> 说明：产品名称中的“训练系列产品”仅表示硬件产品系列。该算子仅支持推理，不支持训练及反向传播。

## 功能说明

- 算子功能：把MLA的KV latent按token压缩成TurboQuant 4bit slot。对每个token先求L2范数并做归一化，再把归一化后的每个元素量化到16个码本中心里最近的一个，得到的4bit索引按维度顺序两两打包进一个字节；范数本身以float16存放在打包区之后，供读取侧还原幅值。

  输入`latent`要求是**已经完成量化基旋转（signed Hadamard）且未归一化**的张量。旋转矩阵本身正交，因此旋转前后的L2范数一致，算子内部重新计算范数不会引入额外误差。

- 计算公式：

  $$
  norm_i = \sqrt{\sum_{d} latent_{i,d}^2 + \epsilon}, \quad \epsilon = 10^{-16}
  $$

  $$
  u_{i,d} = latent_{i,d} \times \frac{1}{norm_i}
  $$

  $$
  nibble_{i,d} = \sum_{b=0}^{14} \left[ u_{i,d} \ge \frac{centroids_b + centroids_{b+1}}{2} \right]
  $$

  $$
  slot_{i,k} = nibble_{i,2k}\ |\ (nibble_{i,2k+1} \ll 4), \quad 0 \le k < headDim/2
  $$

  由于`centroids`升序排列，相邻中心的中点构成15个单调递增的边界，统计`u`超过了多少个边界即等价于取最近的码本中心下标。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
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
      <td>latent</td>
      <td>输入</td>
      <td><ul><li>表示待压缩的KV latent，对应公式中的`latent`。shape为[numTokens, headDim]，仅支持2维。</li><li>要求已完成signed Hadamard旋转且未归一化。</li><li>`headDim`当前仅支持512。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>centroids</td>
      <td>输入</td>
      <td><ul><li>表示4bit量化码本，对应公式中的`centroids`。元素总数必须为16。</li><li>必须按升序排列，否则中点边界不单调，量化结果不可预期。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>slot</td>
      <td>输出</td>
      <td><ul><li>表示压缩后的slot，对应公式中的`slot`。shape为[numTokens, slotSize]，其中slotSize = ceil((headDim / 2 + 2) / 64) * 64，headDim为512时slotSize为320。</li><li>每个slot的布局为：[0, headDim/2)存放打包后的4bit索引，低位nibble对应偶数维；[headDim/2, headDim/2+2)存放float16的`norm`；其余字节为0填充。</li></ul></td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 该接口仅支持推理，不支持训练及反向传播。
- `latent`的`headDim`当前仅支持512（MLA的kv_lora_rank）。slot布局的推导本身是按`headDim`泛化的，放开其他取值需要先补充硬件验证。
- `centroids`元素总数必须为16：4bit索引与16个码本中心是一一对应关系，取值数量不可配置。
- `centroids`必须升序排列，算子不对其做排序或校验。
- 归一化与Hadamard旋转的先后顺序不影响结果，但算子只接受旋转之后的张量；旋转矩阵与码本的生成不在本算子范围内。
- 量化索引取"最近的码本中心"，平局（取值恰好落在两个中心的中点上）时取**较大**的索引。
- 归一化后的取值超出码本首末中心的范围时按最近邻饱和到边界桶，不做截断报错。码本按`N(0, 1/headDim)`拟合，输入分布显著偏离时量化误差会增大，但行为是确定的。

### 非有限输入与范数溢出

算子不对非有限输入做特殊处理，其行为由归一化流程自然导出，且是确定的：

| 输入情况 | `slot`中的量化索引 | 存储的float16范数 |
| :--- | :--- | :--- |
| 该token含NaN | 全部为0（NaN参与的比较恒为false） | NaN |
| 该token含+INF或-INF | INF所在维度为0，其余维度为8 | +INF |
| 全部有限，但`\|\|z\|\|`超出float16表示范围（>65504） | 正常量化 | **+INF，幅值不可恢复** |

最后一行是使用时需要注意的：`slot`只用2字节float16存放范数，若输入张量的L2范数超过65504，读取侧无法还原幅值。MLA的KV latent经RMSNorm后范数在1附近，不会触及该边界。

## 性能

测试环境：Atlas A2 训练系列产品（910B4，40 个 Vector 核，UB 192 KB），CANN 9.0.1。耗时取 `torch_npu.profiler` 落盘的 `kernel_details.csv` 中 `Duration(us)` 的最小值（warmup 5 次、active 10~15 次）。`headDim` 固定为 512。

### 与 DynamicQuant（INT4）的对比

`DynamicQuant` 在 `dst_type` 取 INT4 时同样是"输入张量进、4bit + scale 出"的 per-token 压缩，是仓内唯一无融合的同类算子，因此作为对照。两者的量化方式不同：`DynamicQuant` 是均匀仿射（per-token max-abs 求 scale 后取整），本算子是 16 级码本的最近邻搜索，后者需要 15 轮串行的比较/选择/累加，计算量本身高一个量级。

| numTokens | TurboQuantCompressLatent | DynamicQuant(INT4) | 倍数 |
| ---: | ---: | ---: | ---: |
| 1 | 4.38 us | 2.76 us | 1.6x |
| 32 | 5.32 us | 3.78 us | 1.4x |
| 128 | 6.72 us | 4.18 us | 1.6x |
| 512 | 12.86 us | 5.64 us | 2.3x |
| 1024 | 20.22 us | 7.60 us | 2.7x |
| 4096 | 64.22 us | 13.70 us | 4.7x |
| 16384 | 242.62 us | 35.84 us | 6.8x |
| 65536 | 979.46 us | 124.76 us | 7.9x |

每 token 搬运字节数：本算子 2368 B（fp32 输入 + 320 B slot），`DynamicQuant` 1284 B（fp16 输入 + INT4 + fp32 scale）。`DynamicQuant` 在大 shape 下已跑在访存带宽上（实测约 384 GB/s），差距来自码本搜索的计算量而非实现效率。

### 多 token 批处理带来的收益

最近邻搜索是 15 轮存在数据依赖的向量指令，每条指令只覆盖约 8 拍的运算却要付约 35 拍的固定发射开销，因此瓶颈是流水线延迟而非吞吐。Tiling 据此把 `tokensPerBatch` 个 token 折进同一条向量指令（`tokensPerBatch = min(tokensPerCore, 12)`，token 数不足时自动退化为 1，不占用额外核）。

下表为同一份二进制、仅将 `tokensPerBatch` 强制为 1 所测得的对照：

| numTokens | tokensPerBatch = 1 | 自适应 tokensPerBatch | 加速比 |
| ---: | ---: | ---: | ---: |
| 1024 | 50.34 us | 20.32 us | 2.48x |
| 2048 | 96.60 us | 35.12 us | 2.75x |
| 4096 | 187.38 us | 64.28 us | 2.92x |
| 8192 | 368.65 us | 124.04 us | 2.97x |
| 16384 | 733.19 us | 242.76 us | 3.02x |

批处理前后输出逐字节一致。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_turbo_quant_compress_latent](examples/test_aclnn_turbo_quant_compress_latent.cpp) | 通过[aclnnTurboQuantCompressLatent](docs/aclnnTurboQuantCompressLatent.md)接口方式调用TurboQuantCompressLatent算子。 |
| PyTorch接口 | [turbo_quant_compress_latent](torch_extension/turbo_quant_compress_latent.py) | 通过`torch.ops.cann_ops_nn.turbo_quant_compress_latent`调用TurboQuantCompressLatent算子。 |
| 图模式 | [graph_convert_turbo_quant_compress_latent](torch_extension/graph_convert_turbo_quant_compress_latent.py)  | 通过[算子IR](op_graph/turbo_quant_compress_latent_proto.h)构图方式调用TurboQuantCompressLatent算子。         |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| chen-weipeng12 | 个人开发者 | TurboQuantCompressLatent | 2026/08/04 | TurboQuantCompressLatent算子AscendC实现，适配开源仓 |
