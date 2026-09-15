# FusedScatterCopySparseFlashAttention

## 产品支持情况

| 产品 | 是否支持 |
|:---|:---:|
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

`fused_scatter_copy_sparse_flash_attention` 面向 GLM MLA/SFA 的稀疏 KV Cache 卸载场景，在一次公开 Torch
调用中完成 DRAM 到 HBM 的缺失 KV 搬运和因果 Sparse Flash Attention。算子支持动态
TND Batch，每个请求可包含 1～16 路 Query，对应 MTP0～MTP15。

对每一路 Query，Attention 使用两部分 KV：

- `topk_dst_slots` 指定的 2048 个 HBM 稀疏 Cache Slot；
- 当前 Query 因果可见、且未卸载的连续 Tail KV。

Attention Score 为：

$$
score=(Q_{nope}K_{nope}^{T}+Q_{rope}K_{rope}^{T})\times scale
$$

算子根据设备侧 Metadata 自动选择执行路径：

- 稳态路径：融合读取 `topk_src_ids`，搬运每一路 Query 的 Miss KV，并执行 Attention；
- 首次填充路径：内部 helper 先按请求级 `miss_*` 清单填充 HBM Cache，再执行只访问 HBM
  的 Attention；
- 路径选择不读取 CPU 数据，可用于 ACLGraph Capture 和 Replay。

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|:---|:---:|:---|:---:|:---:|
| query_rope | 输入 | TND RoPE Query，Shape 为 `(T,N,64)`。 | FLOAT16、BFLOAT16 | ND |
| query | 输入 | TND NoPE Query，Shape 为 `(T,N,512)`。 | FLOAT16、BFLOAT16 | ND |
| actual_seq_lengths_query | 输入 | 各请求 Query 在 T 维的累计结束位置，Shape 为 `(B)`。 | INT32 | ND |
| actual_seq_lengths_kv | 输入 | 各请求最后一路 Query 的 HBM 逻辑 KV 长度，Shape 为 `(B)`。 | INT32 | ND |
| num_cache_tokens | 输入 | 各请求 HBM 稀疏 Cache 容量，Shape 为 `(B)`。 | INT32 | ND |
| topk_dst_slots | 输入 | 每路 Query 的 TopK HBM Logical Slot，Shape 为 `(T,1,2048)`。 | INT32 | ND |
| topk_src_ids | 输入 | 与 `topk_dst_slots` 对应的 Source Token ID，Shape 为 `(T,1,2048)`。 | INT32 | ND |
| topk_miss_counts | 输入 | 每路 Query 在 TopK 中的 Miss 前缀长度，Shape 为 `(T)`。 | INT32 | ND |
| miss_src_ids | 输入 | 请求级去重后的首次填充 Source Token，Shape 为 `(B,32768)`。 | INT32 | ND |
| miss_dst_slots | 输入 | 与 `miss_src_ids` 对应的 HBM Logical Slot，Shape 为 `(B,32768)`。 | INT32 | ND |
| miss_counts | 输入 | 每个请求有效的首次填充搬运项数，Shape 为 `(B)`。 | INT32 | ND |
| hbm_block_table | 输入 | HBM 逻辑块到物理块的映射，Shape 为 `(B,hbm_max_blocks)`。 | INT32 | ND |
| dram_block_table | 输入 | DRAM 逻辑块到物理块的映射，Shape 为 `(B,dram_max_blocks)`。 | INT32 | ND |
| hbm_k_rope | 输入/输出（原地更新） | HBM KPE Cache，Shape 为 `(hbm_blocks,128,1,64)`。 | FLOAT16、BFLOAT16 | ND |
| hbm_kv_cache | 输入/输出（原地更新） | HBM CKV Cache，Shape 为 `(hbm_blocks,128,1,512)`。 | FLOAT16、BFLOAT16 | ND |
| dram_k_rope | 输入 | DRAM KPE Cache，Shape 为 `(dram_blocks,128,64)`。 | FLOAT16、BFLOAT16 | ND |
| dram_kv_cache | 输入 | DRAM CKV Cache，Shape 为 `(dram_blocks,128,512)`。 | FLOAT16、BFLOAT16 | ND |
| scale_value | 输入 | Attention Score 的缩放系数。 | FLOAT | - |
| attention_out | 输出 | 调用方预分配的 Attention 结果，Shape 为 `(T,N,512)`。 | FLOAT16、BFLOAT16 | ND |

其中，B 为请求数，T 为所有请求的 Query 总数，N 为 Query Head 数。公开接口返回值为
`None`，结果写入 `attention_out`，同时按需原地更新两个 HBM Cache Tensor。

## MTP Query 排布

对请求 `i`：

```text
query_start = i == 0 ? 0 : actual_seq_lengths_query[i - 1]
query_end   = actual_seq_lengths_query[i]
Q           = query_end - query_start
```

Q 为该请求的 Query 路数，范围为 1～16。区间内第 `query_row` 路 Query 的因果可见 KV
长度为：

```text
visible_kv_len = actual_seq_lengths_kv[i] - (query_end - 1 - query_row)
```

`actual_seq_lengths_query` 必须严格递增，最后一项必须等于 T。

## 首次填充与稳态路径

当任一请求满足 `miss_counts[i] >= num_cache_tokens[i]` 时，本 Batch 进入首次填充路径。
内部 `fused_scatter_copy_sparse_flash_attention_first_fill` helper 按每个请求自己的 `miss_counts` 有效前缀完成搬运，
然后主 Kernel 使用 `topk_dst_slots` 从 HBM 计算 Attention。

其他 Batch 进入稳态路径。每一路 Query 的前 `topk_miss_counts[row]` 个 TopK 项视为 Miss，
主 Kernel 使用对应的 `topk_src_ids` 从 DRAM 读取并写入 `topk_dst_slots`，同时与 Attention
流水重叠；TopK 后缀为已经驻留 HBM 的 Hit。

## 约束说明

- 仅支持推理场景，并支持图模式。
- 所有 Tensor 必须位于同一 NPU、保持连续且不能为空。
- 所有浮点 Tensor 必须使用相同数据类型，只支持 FLOAT16 或 BFLOAT16。
- 所有 Metadata Tensor 必须为 INT32。
- Block Size 固定为 128，TopK 固定为 2048，KV Head 数固定为 1。
- N 支持 8、16、32、64、128；Head Dim 固定为 512，RoPE Dim 固定为 64。
- 必须满足 `B>0`、`B<=T<=16B`，每个请求的 Query 路数为 1～16。
- `miss_src_ids` 和 `miss_dst_slots` 的宽度固定为 32768，即 `16×2048`，可容纳
  MTP15 下各路 TopK Miss 完全不重合的最坏情况。
- HBM/DRAM Block Table 必须覆盖所有实际访问的 Logical Block，活动 Source ID 和 Slot
  不能为负数。
- `attention_out` 的 Shape 和数据类型必须与 `query` 一致。

## 调用示例

该算子通过 vLLM Ascend 的 Torch 扩展调用，调用方负责预先分配输出 Tensor：

```python
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

attention_out = torch.empty_like(query)
torch.ops._C_ascend.npu_fused_scatter_copy_sparse_flash_attention(
    query_rope=query_rope,
    query=query,
    actual_seq_lengths_query=actual_seq_lengths_query,
    actual_seq_lengths_kv=actual_seq_lengths_kv,
    num_cache_tokens=num_cache_tokens,
    topk_dst_slots=topk_dst_slots,
    topk_src_ids=topk_src_ids,
    topk_miss_counts=topk_miss_counts,
    miss_src_ids=miss_src_ids,
    miss_dst_slots=miss_dst_slots,
    miss_counts=miss_counts,
    hbm_block_table=hbm_block_table,
    dram_block_table=dram_block_table,
    hbm_k_rope=hbm_k_rope,
    hbm_kv_cache=hbm_kv_cache,
    dram_k_rope=dram_k_rope,
    dram_kv_cache=dram_kv_cache,
    scale_value=scale_value,
    attention_out=attention_out,
)
```

Python 接口内部依次调用私有首次填充 helper 和主算子的 ACLNN 两段式接口，调用方无须
自行申请 Workspace 或管理 `aclOpExecutor`。

## 测试说明

正确性测试位于
`tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_scatter_copy_sparse_flash_attention.py`，需要在已经
编译安装当前代码的 Ascend NPU 环境执行：

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_scatter_copy_sparse_flash_attention.py
```

测试覆盖以下场景：

- FLOAT16、BFLOAT16 和 8/16/32/64/128 个 Query Head；
- MTP0、异构 MTP2/MTP3/MTP8/MTP15 动态 TND Batch；
- 稳态融合搬运、首次填充及混合请求 Batch；
- `attention_out` 与 CPU Golden 的精度对比以及 HBM Cache 搬运结果；
- ACLGraph Capture/Replay，以及 Capture 稳态分支后由设备 Metadata 切换到首填分支。

## 典型时延

下表为 Ascend 910_93、BF16、8 个 Query Head、MTP3、Source Length=65536、HBM Cache
容量=12288、Tail=64、每请求约 300 个 Unique Miss 的单算子实测数据。测试包含 10 次
预热和 200 次计时，时延单位为 μs。

| Batch Size | ScatterCopy | SFA-MTP | Fused | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 41.0 | 174.9 | 193.5 | 1.151× |
| 12 | 53.5 | 202.6 | 209.8 | 1.160× |
| 16 | 68.9 | 277.1 | 321.9 | 1.057× |
| 24 | 97.8 | 418.0 | 426.6 | 1.235× |

时延会随硬件、CANN 版本、编译选项和运行负载变化，表中结果仅说明该配置下的典型表现，
不作为性能承诺或测试通过阈值。
