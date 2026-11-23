# QuantLightningIndexerV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR&950DT 系列产品</term>|      √     |
|<term>Atlas A3 系列产品</term>|      √     |
|<term>Atlas A2 系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- API功能：QuantLightningIndexerV2是推理场景下，稀疏attention前处理的计算，选出关键的稀疏token，并对输入query和key进行量化实现存8算8，获取最大收益。

- 计算公式：
    $$out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\}$$
    主要计算过程为：
    1. 将某个token对应的输入参数`query`（$Q_{index}^{Quant}\in\R^{g\times d}$）乘以给定上下文`key`（$K_{index}^{Quant}\in\R^{S_{k}\times d}$），得到相关性。
    2. 相关性结果与`query`和`key`对应的反量化系数`query_dequant_scale`（$Scale_Q$）和`key_dequant_scale`（$Scale_K^T$）相乘，通过激活函数$ReLU$过滤无效负相关信号后，得到当前Token与所有前序Token的相关性分数向量。
    3. 将其与权重系数`weights`（$W$）相乘后，沿g的方向，选取前$Top-k$个索引值得到输出$out$，作为Attention的输入。

## 参数说明

| 参数名                     | 输入/输出/属性 | 描述  | 数据类型       | 数据格式   |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| query                     | 输入      | 公式中的$Q_{index}^{Quant}\in\R^{g\times d}$，表示输入Index Query，不支持非连续。| INT8、FLOAT8_e4m3fn、HIFLOAT8、FLOAT4_e2m1 | ND         |
| key                   | 输入      | 公式中的$K_{index}^{Quant}\in\R^{S_{k}\times d}$，表示压缩后的输入Index Key，支持0轴非连续。| INT8、FLOAT8_e4m3fn、HIFLOAT8、FLOAT4_e2m1 | ND |
| weights                 | 输入      | 公式中的$W$，表示权重系数，不支持非连续。 | FLOAT16、FLOAT32 | ND |
| query_dequant_scale             | 输入      | 公式中的$Scale_Q$，表示Index Query的反量化系数，不支持非连续。`quant_mode`为3/5时，shape为将`query`的D轴替换为(D/64, 2)；`quant_mode`为4时，shape为(1,)；其他场景shape与weights一致 | FLOAT16、FLOAT32、FLOAT8_e8m0     | ND         |
| key_dequant_scale            | 输入      | 公式中的$Scale_K$，表示Index Key的反量化系数，支持0轴非连续。`quant_mode`为3/5时，shape为将`key`的D轴替换为(D/64, 2)；`quant_mode`为4时，shape为(1,)；其他场景shape为移除`key`的D轴 | FLOAT16、FLOAT32、FLOAT8_e8m0       | ND         |
| cu_seqlens_q                    | 可选输入      | layout_q为TND时必须传入，表示每个Batch中`query`的有效token数前缀和。；layout_q为BSND时不能传入 | INT32       | ND         |
| cu_seqlens_k                    | 可选输入      | layout_k为TND时必须传入，表示每个Batch中`key`的有效token数前缀和；layout_k为PA_BSND或BSND时不能传入 | INT32       | ND         |
| seqused_q                    | 可选输入      | layout_q为BSND时可选传入，表示每个Batch中`query`的有效token数 | INT32       | ND         |
| seqused_k                    | 可选输入      | layout_k为PA_BSND或BSND时使用，表示每个Batch中`key`的有效token数。| INT32       | ND         |
| cmp_residual_k                    | 可选输入      | 压缩场景下Key的残余长度，需满足0 \<= cmp_residual_k\[i\] \< cmp_ratio。| INT32       | ND         |
| block_table                    | 可选输入      | 表示PageAttention中KV存储使用的block映射表。 | INT32       | ND         |
| output_idx_offset                    | 可选输入      | 输出索引的偏移量 | INT32       | ND         |
| metadata                    | 可选输入      | QuantLightningIndexerV2Metadata算子传入的分核信息，包含使用核数、分块大小以及每个核处理数据的起始点等内容。 | INT32       | ND         |
| causal_seqlens_kv_global    | 可选输入      | DCP（Decode Context Parallel，解码上下文并行）场景下每个请求本次 Q 片段的全局 exclusive 上界 C（最后一个 Q 行的全局位置 + 1）；非 batch 前缀和，不是物理 cache 容量。`enable_dcp` 为 true 时必传 | INT32       | ND         |
| quant_mode                 | 属性      | 用于标识输入的量化模式。 | INT32          | -         |
| max_seqlen_q                 | 可选属性| Query的最大序列长度，默认值-1表示任意可能长度 | INT32 | -         |
| layout_q                 | 可选属性| 用于标识输入`query`的数据排布格式，默认值"BSND"。 | STRING | -         |
| layout_k      | 可选属性      | 用于标识输入`key`的数据排布格式，默认值"BSND"。| STRING          | -         |
| topk  | 属性      | 代表topK阶段需要保留的索引数量，默认值2048。 | INT32          | -         |
| mask_mode | 可选属性      | 表示mask的模式，默认值0。 | INT32          | -         |
| cmp_ratio      | 可选属性      | 用于稀疏计算，表示key的压缩倍数，默认值1。 | INT32          | -         |
| return_value      |  可选属性     | 表示是否输出`sparse_values`，默认值0。DCP 场景要求为1（跨 rank 全局 TopK 需要 score）。 | INT32          | -         |
| enable_dcp      |  可选属性     | 是否开启 DCP（Decode Context Parallel，解码上下文并行）位置语义，默认值false。 | BOOL          | -         |
| cp_world      |  可选属性     | CP 组大小 P，默认值1。 | INT32          | -         |
| cp_rank      |  可选属性     | 当前 rank r，默认值0，要求 0 ≤ r < cp_world。 | INT32          | -         |
| cp_kv_cache_interleave_size      |  可选属性     | 每 S 个 token 轮换 rank，默认值1；逐 token 分片为1，整 block 分片时传物理 block_size。 | INT32          | -         |
| sparse_indices     | 输出      | 公式中的输出Out，参与稀疏attention计算的token索引值。 | INT32          | ND         |
| sparse_values           | 输出      | 公式中的Indices输出对应的value值。`return_value`为1时shape与`sparse_indices`一致，`return_value`为0时shape为(0,) | BFLOAT16         | ND          |

## 约束说明

- <term>Ascend 950PR&950DT 系列产品</term>：
    - `query`、`key`在`quant_mode`为1/3时支持FLOAT8_e4m3fn，`quant_mode`为2时支持INT8，`quant_mode`为4时支持HIFLOAT8，`quant_mode`为5时支持FLOAT4_e2m1。
    - `query_dequant_scale`和`key_dequant_scale`在`quant_mode`为1/4时支持FLOAT32，`quant_mode`为2时支持FLOAT16，`quant_mode`为3/5时支持FLOAT8_e8m0。
    - `weights`在`quant_mode`为2时支持FLOAT16，`quant_mode`为1/3/4/5时支持FLOAT32。
    - `quant_mode`为3/5时，`query_dequant_scale`和`key_dequant_scale`的维数分别比`query`和`key`多1，前缀维度保持一致，末两维为(D/64,2)；D必须为64的倍数，每个scale对应连续32个D轴逻辑元素。
    - `query`的N支持[1, 64]，`key`的N仅支持1。
    - `topk`支持[1, 8192]。
    - 当传入的参数layout_query为TND时，必须传入cu_seqlens_q，如果也传入seqused_q，应保证由seqused_q传入的各个batch的query长度不超过根据cu_seqlens_q计算出的各个batch的q序列长度。当某个batch由seqused_q传入的q序列长度seqlen1小于由cu_seqlens_q计算出的query长度seqlen2时，会启用TND Padding功能，将该batch的seqlen2与seqlen1差值部分的query输出的sparse_indices和sparse_values全部置为无效值。部分长序列场景下，如果需要填充的无效数据过多，由于硬件限制可能会导致aicore执行超时，可以通过(seqlen2 - seqlen1) * topk来计算需要填充的数据量，建议将这个数据量控制在4亿以内。

- <term>Atlas A3 系列产品</term>、<term>Atlas A2 系列产品</term>：
    - `quant_mode`仅支持2。
    - `query`、`key`支持INT8，不支持FLOAT8_e4m3fn、HIFLOAT8和FLOAT4_e2m1。
    - `query_dequant_scale`和`key_dequant_scale`支持FLOAT16，不支持FLOAT32和FLOAT8_e8m0。
    - `weights`支持FLOAT16，不支持FLOAT32。
    - 不支持`output_idx_offset`；`return_value` 支持 0/1（DCP 跨 rank 全局 TopK 需要 score，DCP 场景强制为1）。
    - `query`的N仅支持64，`key`的N仅支持1。
    - `topk`支持[1, 2048]。

## DCP（Decode Context Parallel，解码上下文并行）

CP 场景下 KV cache 按 token 轮转切分到各卡，算子不再需要框架构造二维 mask，而是由 `enable_dcp` + `cp_world`/`cp_rank`/`cp_kv_cache_interleave_size` + `causal_seqlens_kv_global` 直接计算每行本 rank 的可见长度：

```text
global_q = C - Q_len + i
global_k = ((local_k // S) * P + r) * S + local_k % S
visible  = (local_k < L) and (global_k <= global_q)
```

其中 `C = causal_seqlens_kv_global[b]`，`L = seqused_k[b]`（本地有效长度），`P/cp_world`、`r/cp_rank`、`S/cp_kv_cache_interleave_size`。

DCP 约束（arch22：Atlas A2/A3；arch35：Ascend950/A5）：

- `layout_q` 必须为 TND，`layout_k` 必须为 PA_BBND；
- `cmp_ratio` 必须为 1（CP 不与 KV 压缩叠加）；
- `candidate_mode` 必须为 3（DCP 与两级 candidate 互斥）；
- `return_value` 必须为 1；
- 不能传 `output_idx_offset`；
- 输出 index 为**本地逻辑 KV 序号**，由框架按 `global = (local // S * P + r) * S + local % S` 转全局后合并 TopK。

A5（arch35）额外要求：`VllmAscendQuantLightningIndexerV2Metadata` 必须同步传入 `causal_seqlens_kv_global` 与
`enable_dcp/cp_world/cp_rank/cp_kv_cache_interleave_size`。A5 的 FD/LD 会把 metadata 输出的 `s2End` 当作权威块上界，
metadata 与 kernel 使用同一套 DCP 可见长度公式，否则会漏算。

DCP 的详细设计文档仅保留在本地，不入库；仓库内的接口、约束与验证结论以本 README 和
[docs/aclnnQuantLightningIndexerV2.md](docs/aclnnQuantLightningIndexerV2.md) 为准。

## 调用示例

| 调用方式 | 调用样例 | 说明 |
|----------|----------|------|
| PyTorch API | - | 通过[torch.ops.cann_ops_transformer.quant_lightning_indexer](../../torch_extension/cann_ops_transformer/docs/zh/quant_lightning_indexer.md)接口调用QuantLightningIndexerV2算子。 |
| aclnn API | [test_aclnn_quant_lightning_indexer_v2](examples/test_aclnn_quant_lightning_indexer_v2.cpp) | 通过[aclnnQuantLightningIndexerV2](docs/aclnnQuantLightningIndexerV2.md)两段式接口调用QuantLightningIndexerV2算子。 |

单算子精度回归位于 `tests/e2e/nightly/single_node/ops/singlecard_ops/test_quant_lightning_indexer_v2.py::test_quant_lightning_indexer_v2_accuracy`，覆盖 A3 INT8、A3 DCP 和 A5 FP8。

## DCP 本地候选入口（Ascend 950 C8）

### Kimi A5 native DCP 框架路径

启用现有 sharded DCP indexer 时，A5 C8 自动使用 `npu_quant_lightning_indexer_v2`
的 `enable_dcp=True` 路径，无需新增环境变量或服务参数。依赖包含 DCP 接口的
QLI V2 和 metadata 算子包，以及重新编译的 vllm-ascend C++ 扩展。

- 保持 FP8 E4M3 query/key、FP32 weights/descale、`quant_mode=1`、
  `mask_mode=3`、`return_value=1`，key/scale 仍可共享非连续物理页。
- metadata producer 和 QLI consumer 使用同一份全局 `causal_seqlens_kv_global`
  及实际 DCP group 的 world/rank/interleave；`seqused_k` 仍是各请求本地有效 KV 长度。
  metadata 在同一 forward 内跨层复用，保留 V2 runner 的提前生成和等待机制。
- 框架只计算请求级本地 KV 长度，逐 query 因果可见范围由 QLI 内部计算。
- 每个 rank 返回完整本地 K 个候选；现有候选打包算子将 BF16 score 转为 FP32
  传输值，再映射全局 ID、交换并合并 TopK。这不提高原始 score 的精度。
  C++ 绑定初始化图 padding 行为 `(-1, -inf)`，防止未写输出进入候选交换。

非 A5、非 sharded DCP 及非 C8 路由不由本次接入切换。
原有私有本地候选接口继续保留，其契约如下。

回归入口：`tests/e2e/nightly/single_node/ops/singlecard_ops/test_a5_qli_native_dcp.py`。
覆盖因果边界、共享页 stride、长 KV、旧路径/CPU 数值对照、图重放和候选合并。
单卡模拟 owner 合并不等于真实 DCP 通信或整网性能验证。

`torch.ops._C_ascend.npu_quant_lightning_indexer_local_candidates` 使用现有 QLI
Cube/Vector 计算和 BF16 派生比较键，输出本地 INT32 索引以及 FP32 传输格式的比较值，
形状均为 `[T, 1, K]`。FP32 转换保留比较值，不将算子的排序精度改称 FP32。
比较键保留现有 `CAST_ROUND` 中点规则（远离零舍入），与 PyTorch BF16 的 RNE 不同。
Query `[T, H, 128]` 和 key `[blocks, block_size, 1, 128]` 为 FP8 E4M3；
weights、query/key descale 为 FP32。每个 rank 必须使用完整相同的 H 个 indexer heads。

`local_visible_lengths[T,1]` 是每个 query 因果可见的本地前缀长度。对于 DCP 大小 R、
rank r、interleave c 和全局可见长度 n，调用方计算：

```text
local_visible = (n // (R*c))*c + clamp(n % (R*c) - r*c, 0, c)
global_id = (local_id // c)*(R*c) + r*c + local_id % c
```

新入口在 Top-K 前限制 score 前缀，不能先选出未来 token 再过滤。它使用私有
`return_value=2` 模式，将已有 `output_idx_offset` 输入槽解释为上述本地可见长度；
该模式要求 `mask_mode=0`、`cmp_ratio=1`、`quant_mode=1`，与旧入口的索引偏移语义隔离。
长度被裁剪至 `[0, seqused_k]`；零长度、未使用的 TND padding 行返回 `(-1, -inf)`。
长度参与设备端执行，支持图重放时改变可见范围。

Key 和 scale 的 leading-axis stride 分别显式传入 native tiling，支持 128 个 key
与 128 个 FP32 scale 共用一个 16896 字节页，不复制历史 KV 为连续张量。
每个 rank 保留完整本地 K 个候选，框架只交换候选并按比较值和全局 token ID 合并。
不能将本地 K 除以 R。相同分数采用较小 token ID 优先的边界选择。

单算子精度回归：`tests/e2e/nightly/single_node/ops/singlecard_ops/test_quant_lightning_indexer_v2.py`。

## DCP 本地候选入口（A3 INT8 C8）

同一私有入口支持 A3 的 INT8 query/key、FP16 weights 和 query/key descale，
使用 `quant_mode=2`、`return_value=2`、`mask_mode=0`、`cmp_ratio=1`、
`candidate_mode=3`，每个 rank 输出完整本地 K 个候选，K 不超过 2048。
可见长度的含义、因果前缀裁剪、无效行以及仅交换候选的契约与上述 A5 入口一致。

A3 直接输出其原有 Sort/MrgSort 使用的 FP32 比较值，分别覆盖普通输出和 LD
分核归并输出；不将这些比较值转成 BF16。保留原有 Cube 的 INT32 QK、
`1/1024` 归一化、FP16 ReLU/weight 中间值及 FP32 归约过程。
A5 的 FP8 和 BF16 派生比较键路径保持原有行为。

沿用独立 leading-axis stride，支持 128 个 INT8 key 与 128 个 FP16 scale
共用一个 16640 字节页。私有模式将 `output_idx_offset` 槽作为可见长度读取，
不会把该值加到普通输出或 LD 输出的本地索引上。

A3 硬件精度回归：`tests/e2e/nightly/single_node/ops/singlecard_ops/test_quant_lightning_indexer_v2.py`
中的 A3 INT8 和 A3 INT8 DCP 参数组，检查分页索引和分数。
当前 A3 metadata 不启用 LD；私有 LD score 输出仅经过源码审查，尚未硬件验证，
不计入上述测试的运行路径证明。保留 PR 原有 metadata 行为，不强制启用 LD。
超过 K 个相同分数时的小 token ID 边界选择是必须通过的硬件验证项；
单卡 DCP 参数组不替代实际多卡通信与端到端验证。

## A3/A5 兼容边界与框架适配说明

- Host 注册沿用 `kv_quant_sparse_flash_attention_vllm` 的结构：默认签名用于
  Arch22，`aicore_config_95` 完整覆盖 Arch35 的输入输出类型。A3 的 score 输出
  在 return_value!=0 时统一为 FP32（DCP score 与本地候选共用），return_value=0 的
  空占位输出仍为 BF16。A5 的
  dtype 列表和 `return_value` 约束保持原样，不依据 INT8 输入推断设备型号。
  `sparse_values` 的类型由调用方张量和产品注册确定，tiling 按架构和模式校验；
  不在共享 InferDataType 中覆盖它。A3 新增的本地长度和完整 score 形状校验
  仅作用于 Arch22 私有模式。
- A3 沿用 `BaseDeviceAdaptor`，不新增适配类，不修改工厂映射。在 Base 的
  `build_indexer_local_metadata`、`indexer_select_local_candidates` 和量化 SFA
  底层封装中增加 A3 INT8 适配。metadata 沿用 producer 预先生成、跨层共享的
  调用约定，直接复用含前导零的 query start 和预生成 metadata，不在每个
  consumer 内重新 padding 或构建 metadata；
  公共 `execute_sparse_flash_attention_process` 和整个 `A5DeviceAdaptor`
  保持发布基线源码。硬件 profile 和 family 枚举同样保留基线，不新增
  `INT8_SPARSE`。现有 `device_config.py` 内的纯函数统一限定 A3 适用范围，
  配置继续使用同一设备层接口。
  A3 INT8 使用 1、2、4、8、16、32、64、128 个 heads：例如 6 补到 8，
  96 补到 128。在量化 SFA 封装内先拼接 NoPE/RoPE，再对完整 query 分组
  padding，避免对零宽 RoPE 张量调用 ConstantPadNd；算子返回后在 DCP merge
  前裁剪 attention/max/sum。A5 的 LSE/NoPE 也会转调这个 Base 封装，因此
  新分支同时检查 A3 profile 和 INT8 KV；A5、其他设备及非 INT8 输入不触发
  新 padding，保留原有分发、FP8 分组规则和结果。
- `AscendConfig` 的 `W8A8_DYNAMIC` 回退只在 A3 显式请求 indexer INT8 cache
  时启用。部分 ModelSlim 检查点只描述 `indexer.wq_b.weight` 的投影量化，
  没有逐层 `indexer.quant_type`；若不识别它，会错误漏掉实际 C8 层。
  A3 显式的 `indexer.quant_type` 优先于投影类型，特别是 FLOAT/BF16 回退层。
  A5 保留基线的类型列表及投影/显式类型解析行为，不套用 A3 回退规则。
- projector rotation 加载不属于本 PR 的 A3 算子适配范围，`kimi_k3.py`
  保留发布基线。旋转关闭但 checkpoint 残留 `mm_projector.rot_proj.weight`
  的场景由独立 PR8 处理；验证该类权重时需叠加 PR8，不根据残留张量
  自动启用旋转，也不修改权重或启用 DSpark。
