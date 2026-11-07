# FusedQuantLightningIndexerManage（A5）

## 产品差异及约束

本文面向 <term>Ascend 950PR&950DT 系列产品</term>（A5）。与 <term>Atlas A3 系列产品</term>、<term>Atlas A2 系列产品</term> 的差异及约束如下；A2/A3的功能、参数及约束说明见 [README](README.md)。

| 项目 | <term>Atlas A3 系列产品</term> / <term>Atlas A2 系列产品</term>（arch22） | <term>Ascend 950PR&950DT 系列产品</term>（arch35） |
| :--- | :--- | :--- |
| Query / Index Key | INT8 | FLOAT8_E4M3FN |
| Index Weights | BFLOAT16 | BFLOAT16 |
| Query / Key dequant scale | FLOAT16 | FLOAT32 |
| 卸载状态 TopK 范围 | 稳定 Prefix `[0,L)` | 每路完整因果可见范围 `[0,V)`，包括 tail |
| 选中 tail 的 Slot | 不适用 | `C + source_id % 256`，固定作为 Hit，不进入 Miss 搬运计划或持久化 Pool |
| 首次卸载每路 Miss 数 | 2048 | 选中的 Prefix Token 数；请求级填充数量仍为 C |
| Prefix 因果约束 | `L <= floor((S-Q)/128)*128` | `L <= S-(Q-1)` 且 `S-L <= 256` |
| 稀疏缓存 C 上限 | 32640 | 32384；调用方另行提供 256 个 tail Slot |
| 非连续输入 | 不支持 | Weights 的 T 轴及 Key / Key scale 的 Block 轴允许 stride，其他轴保持连续 |
| 非卸载输出顺序 | Score 排序 | 保留官方 A5 histogram TopK 的收集顺序，不保证全局 Score 降序 |

其中S为本轮实际KV长度，Q为请求的Query路数，L为卸载Prefix长度，C为持久化稀疏缓存容量，V为当前Query的因果可见长度。非卸载`-3`状态下，两种架构均在各路完整因果可见范围内检索。

A5要求L、C为128的倍数，L至少为2048，C不超过L。当`L<=Q*2048`时要求`C=L`，否则要求`C>=Q*2048`；无论哪种情况，C均不得超过32384。调用方另外提供256个tail Slot。

两种架构均支持Q=1～14、N=32/64、Head Dim=128、Block Size=128，Source容量最大为`2^21`，Miss输出容量为32768。

A5在FP32域累加FP8点积、折叠Weights与Query scale，聚合后乘Key scale，并将最终分数转为BF16用于TopK。A2/A3使用INT32点积、FP16中间转换和第二段MMA。两种架构的中间精度及淘汰候选选择不同，不要求跨架构TopK、回收Slot或Pool更新逐元素一致；同分截断位置的具体Token选择也不作为跨架构一致性契约。

## 功能说明

- 算子功能：`fused_quant_lightning_indexer_manage`的A5实现是量化（C8）Lightning Indexer与稀疏KV Cache管理的融合算子。Index Query和Index Key Cache使用FLOAT8_E4M3FN。在一次NPU调用中完成Top-2048检索、HBM稀疏缓存初始化或稳态淘汰、MTP多路Query的TopK Union管理，以及Cache Miss搬运计划生成。一个Batch可以混合非卸载、首次卸载和稳态卸载请求，每个请求支持1～14路Query。

- 量化Lightning Indexer的计算关系：

  $$
  Indices=\operatorname{TopK}_{2048}\left(\operatorname{bf16}\left[\left(W\odot q_{scale}\right)^T@\operatorname{ReLU}\left(Q^{q}_{index}@K^{qT}_{index}\right)\odot k_{scale}\right]\right)
  $$

  其中，$Q^{q}_{index}$为某一路Query的FLOAT8_E4M3FN量化Index Query，shape为$(N,128)$；$K^{q}_{index}$为该路Query完整因果可见范围内的量化Index Key，shape为$(V,128)$；$W$为该路Query的BFLOAT16聚合权重，按$(N,1)$列向量表示；$q_{scale}$、$k_{scale}$为FLOAT32反量化Scale，分别采用per-(token,head)和per-token粒度。点积及加权聚合在FP32域进行，最终分数转为BF16后进入histogram TopK。公式表达计算关系，浮点中间结果受实际运算次序和舍入影响。

- 稀疏缓存管理：`cache_slots_pool`维护Prefix Source Token到持久化HBM Logical Slot的映射。tail参与TopK，但作为固定Hit使用额外256个循环Slot。算子输出Prefix中本轮需要从CPU搬入HBM的Source Token ID、目标Slot及搬运数量，实际数据搬运和tail KV维护由上层框架执行。

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :---: | :--- | :---: | :---: |
| index_weights | 输入 | 每条Query、每个Index Head的聚合权重，shape为(T,N)，N支持32或64。 | BFLOAT16 | ND |
| query_dequant_scale | 输入 | Query反量化Scale，shape为(T,N)，per-(token,head)粒度，参与Kernel内计算（与`index_weights`折叠）。 | FLOAT32 | ND |
| query | 输入 | TND排布的量化Index Query，shape为(T,N,128)。同一请求的多路Query在T维连续。 | FLOAT8_E4M3FN | ND |
| index_key_dequant_scale | 输入 | Index Key反量化Scale，shape为(index_block_num,128,1)，per-token粒度，经Block Table Gather后参与Kernel内计算。 | FLOAT32 | ND |
| index_key_cache | 输入 | PageAttention排布的量化Index Key Cache，shape为(index_block_num,128,1,128)。 | FLOAT8_E4M3FN | ND |
| index_block_table | 输入 | 每个请求的Index Key逻辑块到物理块映射表，shape为(B,index_max_blocks)，其中index_max_blocks不超过16384。 | INT32 | ND |
| actual_seq_lengths_query | 输入 | 各请求Query在T维的累计结束位置，shape为(B)。元素严格递增，最后一个元素必须等于T。 | INT32 | ND |
| actual_seq_lengths_key | 输入 | 每个请求最后一路Query对应的实际KV长度，shape为(B)。其他路Query的因果可见长度由该值和Query位置推导。 | INT32 | ND |
| offload_seq_lengths_key | 输入 | 每个请求可卸载的稳定Source Prefix长度L；TopK仍检索各路完整因果可见范围，shape为(B)，仅`request_state`为-2或-1时使用。 | INT32 | ND |
| num_cache_tokens | 输入 | 每个请求的持久化HBM稀疏缓存容量C，不包含调用方额外提供的256个tail Slot，shape为(B)，仅`request_state`为-2或-1时使用。 | INT32 | ND |
| request_state | 输入 | 请求状态，shape为(B)。仅支持-3（非卸载）、-2（首次卸载）和-1（稳态卸载）。 | INT32 | ND |
| req_pool_entries | 输入 | 每个请求占用的`cache_slots_pool`行号，shape为(B)。同一次调用中的有效请求必须使用不同的行号。 | INT32 | ND |
| cache_slots_pool | 输入/输出（原地更新） | 跨Decode Step持久化的Source Token到HBM Logical Slot映射，shape为(pool_size,source_capacity)，其中source_capacity=index_max_blocks*128。 | INT32 | ND |
| topk_src_ids | 输出 | 每一路Query的TopK Source Token ID，shape为(T,1,2048)。可见长度不足2048时使用-1填充。 | INT32 | ND |
| topk_dst_slots | 输出 | 与`topk_src_ids`逐项对应的HBM Logical Slot，shape为(T,1,2048)；无有效Slot时为-1。 | INT32 | ND |
| topk_miss_counts | 输出 | 每一路Query在`topk_src_ids`中的Miss前缀长度，shape为(T)；非卸载状态固定为0；卸载状态下选中的tail固定为Hit。 | INT32 | ND |
| miss_src_ids | 输出 | 每个请求本轮需要搬入HBM的Source Token ID，shape为(B,32768)，仅前`miss_counts[i]`项有效。 | INT32 | ND |
| miss_dst_slots | 输出 | 与`miss_src_ids`逐项对应的目标HBM Logical Slot，shape为(B,32768)，仅前`miss_counts[i]`项有效。 | INT32 | ND |
| miss_counts | 输出 | 每个请求的有效搬运项数量，shape为(B)。 | INT32 | ND |

其中，B表示Batch中的请求数量，T表示所有请求的Query总数，N表示Index Head数量。所有Tensor均不支持空Tensor。`index_weights`允许T轴stride，`index_key_cache`和`index_key_dequant_scale`允许Block轴stride；其他输入、输出必须连续。具体步长要求见“约束说明”。

## MTP Query排布

对于请求`i`：

```text
query_start = i == 0 ? 0 : actual_seq_lengths_query[i - 1]
query_end   = actual_seq_lengths_query[i]
Q           = query_end - query_start
```

Q为该请求的Query路数，取值范围为1～14。`actual_seq_lengths_key[i]`表示最后一路Query的KV长度；区间内第`query_row`路Query的因果可见长度为：

```text
visible_length = actual_seq_lengths_key[i] - (query_end - 1 - query_row)
```

三种状态下，每一路Query均在各自的`[0,visible_length)`范围内检索。卸载状态下，`L=offload_seq_lengths_key[i]`将该范围分为稳定Prefix `[0,L)`和tail `[L,visible_length)`；Prefix必须对请求的全部Query因果可见。

选中的Prefix Token参与持久化稀疏缓存管理；选中的tail Token固定为Hit，其Logical Slot为`C + source_id % 256`。调用方负责在额外的256个tail Slot中维护对应KV数据。算子不为tail生成Miss搬运计划，也不将tail写入持久化Pool。

## 请求状态与输出

### 非卸载状态（`request_state=-3`）

- 每一路Query在`[0,visible_length)`范围内执行量化Lightning Indexer。
- `topk_src_ids`保留官方A5 histogram TopK的收集顺序，不保证全局Score降序；`topk_dst_slots`与`topk_src_ids`逐项相同，表示Identity Slot。可见长度不足2048时，两个输出的无效后缀均使用-1填充。
- `topk_miss_counts`和`miss_counts`均为0，`miss_src_ids`和`miss_dst_slots`中的内容无效。
- 每次调用将对应完整`cache_slots_pool`行恢复为Identity Mapping：`pool[row,source]=source`。该状态不使用卸载态的tail Slot规则。

### 首次卸载状态（`request_state=-2`）

- 每一路Query在`[0,visible_length)`内检索，包含tail；忽略对应Pool行的旧映射。
- 算子对各路选中的Prefix Token计算去重Union，再按Source ID递增补充不在Union中的Prefix Token，直至Resident Token数量达到C。
- 算子清空并重建对应Pool行，为C个Resident Prefix Token分配Logical Slot `[0,C)`，其他Source使用`INT32_MIN`标记。
- `miss_src_ids[i,:C]`保存本次需要填充的Resident Prefix Source Token，`miss_dst_slots[i,:C]`为`[0,1,...,C-1]`，`miss_counts[i]=C`。
- 每一路TopK中的Prefix Token均为Miss，tail均为Hit。输出由Miss前缀和Hit后缀组成，各区间按Source ID递增；`topk_miss_counts`等于该路选中的Prefix Token数，不一定为2048。
- tail的Slot为`C + source_id % 256`，不进入Miss搬运计划或持久化Pool。

### 稳态卸载状态（`request_state=-1`）

- 正常Pool行在`[0,L)`中包含C个Resident Source Token，Slot构成`[0,C)`的双射；非Resident Source使用`INT32_MIN`表示，`[L,source_capacity)`不存储持久化映射。
- 每一路Query在完整因果可见范围内检索。Prefix根据Pool判断Hit/Miss，tail固定为Hit并使用`C + source_id % 256`。
- 算子完成Prefix的多路Miss Union、受TopK保护的Victim选择、Slot回收及Pool更新；tail不参与淘汰和持久化管理。
- 每一路TopK输出由Miss前缀和Hit后缀组成，两个区间分别按Source ID递增；`topk_miss_counts`表示Prefix Miss数量。
- `miss_src_ids[i,:miss_counts[i]]`是各路Prefix Miss Source的有序去重Union，`miss_dst_slots`保存这些Source获得的回收Slot。
- 输入Pool行来自非卸载Identity Mapping时，算子保留`[0,C)`并将后缀归一化为`INT32_MIN`，再执行稳态更新。正常稳态调用不会重复扫描完整Pool行。

除-3、-2和-1之外的`request_state`均不支持。

## 约束说明

- 该接口仅支持推理场景，并支持图模式。
- `query`和`index_key_cache`仅支持FLOAT8_E4M3FN，`index_weights`仅支持BFLOAT16，`query_dequant_scale`和`index_key_dequant_scale`仅支持FLOAT32且参与计算。其他输入及所有输出为INT32。
- N支持32或64，Index Head Dim固定为128，KV Head数量固定为1，Block Size固定为128。
- 静态shape必须满足：`B>0`、`B<=T<=14B`、`pool_size>0`、`index_block_num>0`、`1<=index_max_blocks<=16384`，`cache_slots_pool`宽度等于`index_max_blocks*128`。
- `topk_src_ids`和`topk_dst_slots`为(T,1,2048)，`topk_miss_counts`为(T)，`miss_src_ids`和`miss_dst_slots`为(B,32768)，`miss_counts`为(B)。输出由调用方预先分配。
- `actual_seq_lengths_query`必须严格递增，最后一个元素等于T；每个请求的Q在[1,14]范围内。
- 每个请求必须满足`Q<=S<=source_capacity`。Key Cache及Block Table必须覆盖实际访问的物理块。
- 对-2和-1状态，令`L=offload_seq_lengths_key[i]`、`C=num_cache_tokens[i]`、`S=actual_seq_lengths_key[i]`，必须满足`L>=2048`、`0<C<=32384`、`C<=L`，L和C均为128的倍数。
- 当`L<=Q*2048`时必须满足`C=L`；当`L>Q*2048`时必须满足`C>=Q*2048`。
- 卸载Prefix必须对全部Query因果可见：`L<=S-(Q-1)`。整个请求的tail范围满足`0<=S-L<=256`，调用方维护额外256个tail Slot，逻辑范围为`[C,C+256)`。
- `req_pool_entries[i]`必须位于`[0,pool_size)`，同一次调用中的有效请求使用不同Pool行。同一请求生命周期内Pool行号保持不变；正常稳态Pool应满足上述Resident/Slot双射约束。
- `source_capacity`最大为`2^21 = 2097152`。超过`2^17 = 131072`时使用长Source路径；所有对外Source ID始终为完整INT32。
- `INT32_MIN`仅表示Pool中的非Resident Source，不作为有效Slot输出。C的32384上限为tail Slot及内部保留值留出空间。
- NPU计算时，所有输入、输出必须位于同一NPU设备。Meta调用仅做静态shape/dtype/stride校验，不执行计算或逐请求动态值校验。
- `index_weights`仅允许T轴stride：`stride(1)=1`、`stride(0)>=N`。Key仅允许Block轴stride，后三轴步长为`(128,128,1)`；Key scale仅允许Block轴stride，后两轴步长为`(1,1)`。三个传入ACLNN的实际元素步长必须为正，其余输入、输出连续。
- 逐请求动态约束由Kernel检查，Host不通过同步读取NPU Tensor校验请求值。动态非法请求不更新其Pool行，在可安全发布的Query范围内将TopK填为-1、相关计数置0，避免覆盖其他有效请求。该保护不能替代调用方的参数校验。

## 生命周期

| 场景 | 状态序列 | 行为 |
| :--- | :--- | :--- |
| 请求从一开始就卸载 | `-2 -> -1 -> ...` | 首次调用重建Resident集合，后续稳态维护同一Pool行。 |
| 非卸载后直接进入稳态卸载 | `-3 -> -1 -> ...` | 算子将Identity Pool行归一化为Sparse Mapping后执行稳态更新。 |
| 非卸载后重新选择Resident Token | `-3 -> -2 -> -1 -> ...` | `-2`忽略旧Identity Mapping并重新建立Resident集合。 |

## 调用示例

该算子通过vLLM Ascend的Torch扩展调用。调用方负责预先分配全部输出Tensor，算子返回值为`None`，计算结果写入`cache_slots_pool`及6个输出Tensor。

```python
import importlib

import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env

# 先设置自定义算子包路径，再加载Torch扩展。
bootstrap_custom_op_env()
importlib.import_module("vllm_ascend.vllm_ascend_C")

# 此处省略输入Tensor的构造，shape和dtype要求见“参数说明”。
topk_src_ids = torch.full((T, 1, 2048), -1, dtype=torch.int32, device="npu")
topk_dst_slots = torch.full_like(topk_src_ids, -1)
topk_miss_counts = torch.zeros(T, dtype=torch.int32, device="npu")
miss_src_ids = torch.full((B, 32768), -1, dtype=torch.int32, device="npu")
miss_dst_slots = torch.full_like(miss_src_ids, -1)
miss_counts = torch.zeros(B, dtype=torch.int32, device="npu")

torch.ops._C_ascend.npu_fused_quant_lightning_indexer_manage(
    index_weights=index_weights,
    query_dequant_scale=query_dequant_scale,
    query=query,
    index_key_dequant_scale=index_key_dequant_scale,
    index_key_cache=index_key_cache,
    index_block_table=index_block_table,
    actual_seq_lengths_query=actual_seq_lengths_query,
    actual_seq_lengths_key=actual_seq_lengths_key,
    offload_seq_lengths_key=offload_seq_lengths_key,
    num_cache_tokens=num_cache_tokens,
    request_state=request_state,
    req_pool_entries=req_pool_entries,
    cache_slots_pool=cache_slots_pool,
    topk_src_ids=topk_src_ids,
    topk_dst_slots=topk_dst_slots,
    topk_miss_counts=topk_miss_counts,
    miss_src_ids=miss_src_ids,
    miss_dst_slots=miss_dst_slots,
    miss_counts=miss_counts,
)
```

Python接口与A2/A3相同，适配层自动取得三个实际元素步长并调用A5的`aclnnFusedQuantLightningIndexerManageWithStridesGetWorkspaceSize`和`aclnnFusedQuantLightningIndexerManageWithStrides`两段式接口。直接调用ACLNN时，必须提供正值`key_stride`、`scale_stride`、`weight_stride`，分别对应Key、Key scale的Block轴步长和Weights的T轴步长。

Torch适配层使用ordered提交并保护输入、输出和Workspace Tensor的异步生命周期。调用方无需自行申请Workspace或管理`aclOpExecutor`。A5同时提供Meta实现，用于不执行NPU计算的静态参数校验。

## 测试说明

A5正确性测试位于`tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_quant_lightning_indexer_manage_a5.py`。需要在Ascend 950环境编译安装当前Torch扩展和自定义CANN算子包后执行：

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_quant_lightning_indexer_manage_a5.py
```

测试脚本显式设置自定义算子包路径并加载`vllm_ascend.vllm_ascend_C`。A5构建列表包含本算子；修改源码后须重新构建安装，确保运行时使用相应的自定义库及ACLNN入口。

脚本定义114个独立pytest用例：108个NPU用例、6个Meta用例。主要覆盖：

- N=32/64，Q=1～14，三种请求状态及混合Batch；
- tail参与TopK、未来Token的因果屏蔽、Prefix Miss与tail Hit及循环Slot规则；
- 首次填充、稳态淘汰、缓存容量扩展、Miss Union、Pool更新及生命周期；
- Weights/Key/Key scale的stride，静态参数校验、动态非法请求隔离及图重放；
- 17-bit/21-bit Source ID路径及长Source边界；
- 84个短序列非卸载用例，覆盖N=32/64、Q=1～14，以及最短合法长度、128、2047；验证因果可见Token、Identity Slot、零Miss及-1填充；
- 原A5短序列套件中的1/1024/2047/2048/2049、跨越TopK边界的Q4、混合长度Batch和非法请求隔离。

测试使用官方A5量化算子`torch_npu.npu_quant_lightning_indexer`作为TopK参考，按测试场景验证有效Source集合、短序列输出、缓存映射及搬运计划。跨架构浮点精度、同分截断或回收Slot的逐元素一致性不属于测试契约。该脚本验证功能，不设置性能阈值。

## 典型时延

下表为64K MTP3（Q=4）稳态卸载场景的实测结果。测试平台为Ascend 950（A5），Query和Key数据类型为FP8 E4M3FN，Index Head数量为32，Source Prefix长度L为65536，实际KV长度S为65604，HBM稀疏缓存容量C为12288，`request_state=-1`。请求级Union Miss按300～500构造，表中各Batch的实际Union Miss均值为308.000～395.812，每路Query Miss均值为82.000～104.406；时延单位为μs。

计时采用eager单算子、NPU Event和迭代时间的算术均值。每次融合调用前恢复初始Pool，恢复操作不计入算子耗时。

其中，Official表示在融合调用前后各测一次官方`torch_npu.npu_quant_lightning_indexer`，再取两次均值的TopK检索时延；Fused表示`fused_quant_lightning_indexer_manage`完成TopK检索和稀疏缓存管理的总时延；Overhead为两者之差。负值表示该次融合调用比官方LI调用更快，不是独立测得的管理阶段耗时。

| Batch Size | Official | Fused | Overhead |
| ---: | ---: | ---: | ---: |
| 1 | 131.977 | 76.854 | -55.123 |
| 4 | 124.798 | 84.483 | -40.316 |
| 8 | 127.314 | 109.831 | -17.484 |
| 16 | 219.519 | 146.912 | -72.607 |
| 32 | 297.765 | 282.703 | -15.063 |

Overhead沿用原日志数值；日志先计算差值再分别舍入，可能与表中两列相减相差0.001 μs。时延会随硬件、CANN版本、编译选项和运行负载变化，表中结果仅用于说明该测试配置下的典型表现，不作为性能承诺或测试通过阈值。
