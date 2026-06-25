# Qwen3.5 GDN + MTP + 图模式 精度问题分析

> 场景：仅 Qwen3.5（线性注意力 GDN 模型）+ MTP（multi-token prediction / speculative decoding）+ 图模式（ACL Graph）三者同时开启时，decode 阶段出现精度问题。
>
> 现象：4 条并发请求时，`num_speculative_tokens=1` 或 `2` 无精度问题；`num_speculative_tokens=3` 开始出现精度问题。

---

## 一、三者交叉的关键代码位置

GDN × MTP × 图模式的唯一汇合点是 `AscendGDNAttentionMetadataBuilder.build()`。

| 关注点 | 文件 : 行 | 作用 |
|---|---|---|
| Ascend GDN AttentionMeta Builder 主入口 | [vllm_ascend/ops/gdn_attn_builder.py:935-1262](vllm_ascend/ops/gdn_attn_builder.py#L935-L1262) | 三者交叉的唯一 `build()` 入口 |
| 图模式 spec-decode fast path（静态 buffer 填充） | [vllm_ascend/ops/gdn_attn_builder.py:1152-1201](vllm_ascend/ops/gdn_attn_builder.py#L1152-L1201) | `use_full_cuda_graph` 时把 spec tensor 拷进静态 buffer 并做 padding |
| 图模式 non-spec decode fast path | [vllm_ascend/ops/gdn_attn_builder.py:1203-1223](vllm_ascend/ops/gdn_attn_builder.py#L1203-L1223) | 同上，普通 decode 路径 |
| Spec fallback host meta 构造 | [vllm_ascend/ops/gdn_attn_builder.py:683-712](vllm_ascend/ops/gdn_attn_builder.py#L683-L712) | 给 `npu_causal_conv1d_custom` 提供 host args |
| 双缓冲池（无清理） | [vllm_ascend/ops/gdn_attn_builder.py:534-596](vllm_ascend/ops/gdn_attn_builder.py#L534-L596) | `_ascend_gdn_*_pool` round-robin，不清理 tail |
| pinned buffer 拷贝（只写前缀） | [vllm_ascend/ops/gdn_attn_builder.py:599-620](vllm_ascend/ops/gdn_attn_builder.py#L599-L620) | `_copy_to_pinned_cpu` 只写 `[:num_elements]` |
| GDN forward 实际消费这些 tensor | [vllm_ascend/ops/gdn.py:380-495](vllm_ascend/ops/gdn.py#L380-L495) | `index_select(0, spec_token_indx)` 与 conv1d custom op |
| Conv1d host args 重算 / padding 对齐 | [vllm_ascend/ops/gdn.py:106-148](vllm_ascend/ops/gdn.py#L106-L148) | `_pad_conv1d_host_args_to_capture` 把 runtime host args 对齐到 capture 时 `cap_x_dim0` |
| Conv1d 图模式参数更新 | [vllm_ascend/ops/gdn.py:151-253](vllm_ascend/ops/gdn.py#L151-L253) | `update_conv1d_graph_params` 每次 replay 前重算 host args |
| ACL Graph replay（不复制 input） | [vllm_ascend/compilation/acl_graph.py:252-274](vllm_ascend/compilation/acl_graph.py#L252-L274) | 类注释明确："does not store persistent buffers or copy any runtime inputs" |
| Qwen3.5 patch（替换 GDN 实现） | [vllm_ascend/patch/worker/patch_qwen3_5.py](vllm_ascend/patch/worker/patch_qwen3_5.py) | 把 GDN 的 `forward`/`get_state_shape`/`get_attn_backend` 替换为 Ascend 版 |
| 上游 `decode_cudagraph_max_bs` 计算 | [vllm/v1/attention/backends/gdn_attn.py:98-105](vllm/v1/attention/backends/gdn_attn.py#L98-L105) | `max_num_seqs * (num_spec + 1)`，被 `max_cudagraph_capture_size` clip |

---

## 二、图模式脏值（stale value）处理分析

ACL Graph 的设计原则是 **外部代码负责把每个 input tensor 写进静态 buffer**，replay 时只调用 `entry.aclgraph.replay()`，不会重新拷贝 input。因此每个静态 buffer 的"尾部"——即 `[runtime_size:]` 的部分——是否被妥善清零/padding，直接决定图模式下是否会出现脏值。

### 2.1 已正确处理 padding 的 tensor（fast path 内）

[vllm_ascend/ops/gdn_attn_builder.py:1160-1201](vllm_ascend/ops/gdn_attn_builder.py#L1160-L1201) 中，下面这些 tensor 的尾部都被填了 sentinel 值：

| Tensor | Padding 值 | 行号 |
|---|---|---|
| `spec_state_indices_tensor[num_spec_decodes:]` | `NULL_BLOCK_ID` | 1160, 1166 |
| `spec_sequence_masks[num_spec_decodes:]` | `False` | 1173 |
| `spec_query_start_loc[num_spec_decodes+1:]` | `spec_num_query_tokens`（末位累计值） | 1194 |
| `num_accepted_tokens[num_spec_decodes:]` | `1` | 1201 |
| `non_spec_state_indices_tensor[num_decodes:]`（non-spec path） | `NULL_BLOCK_ID` | 1209, 1215 |
| `non_spec_query_start_loc[num_decodes+1:]` | `non_spec_num_query_tokens` | 1223 |

这些是干净的。

### 2.2 ⚠️ 隐患 1：`spec_token_indx` / `non_spec_token_indx` 尾部不清零

[vllm_ascend/ops/gdn_attn_builder.py:1175-1186](vllm_ascend/ops/gdn_attn_builder.py#L1175-L1186)：

```python
self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
    non_spec_token_indx,
    non_blocking=True,
)
non_spec_token_indx = self.non_spec_token_indx[: non_spec_token_indx.size(0)]

self.spec_token_indx[: spec_token_indx.size(0)].copy_(
    spec_token_indx,
    non_blocking=True,
)
spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]
```

问题：

- 只按 runtime size 拷贝，**tail 完全不清零**。
- `self.spec_token_indx` 的实际容量是 `decode_cudagraph_max_bs * (num_spec + 1)`（MTP=3 时 = 64），但只写了前若干个。
- 如果某次迭代 `spec_token_indx.size(0)` 比上次小（例如 request 被 evict、某次 batch 缩小），tail 保留上一轮的索引。
- 下游 [vllm_ascend/ops/gdn.py:414](vllm_ascend/ops/gdn.py#L414) `mixed_quv.index_select(0, spec_token_indx)` 用脏索引去选 mixed_qkv 的行 → conv1d 输入被污染 → 精度问题。
- 图模式 replay 时使用的是 capture 时记录的 shape，**运行时 slice 的 size(0) 不影响 replay 读取的范围**——这反而放大了风险：replay 始终按 capture 时的 size 读取，但 buffer 内容是部分更新的。

**这是最可疑的脏值点。**

### 2.3 ⚠️ 隐患 2：双缓冲池 round-robin 不清零

[vllm_ascend/ops/gdn_attn_builder.py:591-596](vllm_ascend/ops/gdn_attn_builder.py#L591-L596)：

```python
def _acquire_spec_causal_conv1d_host_slot(builder):
    pool = builder._ascend_gdn_spec_causal_conv1d_host_pool
    builder._ascend_gdn_spec_causal_conv1d_host_pool_idx = (
        builder._ascend_gdn_spec_causal_conv1d_host_pool_idx + 1
    ) % len(pool)
    return pool[builder._ascend_gdn_spec_causal_conv1d_host_pool_idx]
```

同样的模式存在于：
- `_ascend_gdn_causal_conv1d_host_pool`（[547-550](vllm_ascend/ops/gdn_attn_builder.py#L547-L550)）
- `_ascend_gdn_chunked_prefill_pool`（[732-735](vllm_ascend/ops/gdn_attn_builder.py#L732-L735)）

`_copy_to_pinned_cpu`（[599-620](vllm_ascend/ops/gdn_attn_builder.py#L599-L620)）只写 `[:num_elements]` 前缀，**tail 保留两轮前的数据**。

幸运的是：消费侧 `to_int64_tuple`（[vllm_ascend/ops/gdn.py:46-50](vllm_ascend/ops/gdn.py#L46-L50)）只对返回的 slice 做 tuple 化，slice 是 `_copy_to_pinned_cpu` 返回的 `pinned_buffer[:num_elements]`，长度正确。所以**当前路径下侥幸正确**，但一旦哪个 consumer 改成读全长（或者 num_elements 在 fallback 路径下取错），立刻爆。

### 2.4 ⚠️ 隐患 3：`extract_hidden_states_proposer.py` 硬编码 num_spec=1

[vllm_ascend/spec_decode/extract_hidden_states_proposer.py:168, 191](vllm_ascend/spec_decode/extract_hidden_states_proposer.py#L168)：

```python
# Since num_speculative_tokens == 1, sampled_token_ids has shape
# (batch_size, 1). For each request we either use the sampled token
# (if valid and not discarded) or a backup token from the request state.
...
# With num_speculative_tokens == 1, there is exactly one token
sampled = sampled_token_ids[:, 0]
is_valid = (sampled >= 0) & (sampled < gpu_input_batch.vocab_size)
valid_sampled_tokens_count = is_valid.to(torch.int32)
```

注释和实现都假设 `num_spec == 1`。如果该 proposer 真的在 num_spec > 1 的路径下跑，返回的 `valid_sampled_tokens_count` 会出错，进而传播到 `num_accepted_tokens`，污染下游所有依赖。

**判定方法**：MTP=2 也走同样的 proposer，如果 MTP=2 没问题，说明这个 proposer 实际上不在 Qwen3.5 + MTP 的链路里（可能是 Eagle 专用，而 Qwen3.5 走的是 `dflash_proposer.py` 或上游 MTP 路径）。需要确认。

### 2.5 ACL Graph replay 行为（背景）

[vllm_ascend/compilation/acl_graph.py:252-274](vllm_ascend/compilation/acl_graph.py#L252-L274) 的类 docstring 明确说：

> "ACLGraphWrapper does not store persistent buffers or copy any runtime inputs into that buffers for replay."

意味着：所有 input tensor 的脏值处理责任完全在 builder 侧（即 [gdn_attn_builder.py](vllm_ascend/ops/gdn_attn_builder.py)）。**builder 没显式清零/padding 的 buffer，replay 时就一定会读到脏值。**

---

## 三、为什么偏偏 MTP=3 出问题（4 条请求）

`decode_cudagraph_max_bs = max_num_seqs * (num_spec + 1)`（[upstream gdn_attn.py:98-105](vllm/v1/attention/backends/gdn_attn.py#L98-L105)）。假设 `max_num_seqs = 4`：

| MTP | 实际 spec tokens（4 reqs） | `decode_cudagraph_max_bs` | vLLM 默认 bucket | fast path 状态 |
|---|---|---|---|---|
| 1 | 8 | 8 | 8 | 正好落在 bucket，无 padding ✅ |
| 2 | 12 | 12 | **16**（向上取整） | fast path，需要补 1 个 dummy seq（`pad_tokens=4, q_per_seq=3`）⚠️ |
| 3 | 16 | 16 | **16**（恰好等于 bucket） | fast path，**无任何 padding 余量** ⚠️ |

### 3.1 假设 A（最可疑）：`max_cudagraph_capture_size` 把 MTP=3 顶出 fast path

[upstream gdn_attn.py:101-105](vllm/v1/attention/backends/gdn_attn.py#L101-L105)：

```python
if self.compilation_config.max_cudagraph_capture_size is not None:
    self.decode_cudagraph_max_bs = min(
        self.decode_cudagraph_max_bs,
        self.compilation_config.max_cudagraph_capture_size,
    )
```

如果 `max_cudagraph_capture_size` 被设到 8 或 12，那么：
- MTP=1 (`max_bs=8`) 和 MTP=2 (`max_bs=12`) 仍可落在 fast path 内；
- MTP=3 (`max_bs=16`) 被 clip 到 < 16，触发 `num_spec_decode_tokens (16) > decode_cudagraph_max_bs`，**直接退出 fast path**（[gdn_attn_builder.py:1156-1157](vllm_ascend/ops/gdn_attn_builder.py#L1156-L1157)），改走 fallback pinned-buffer 路径。

慢路径里 `_build_spec_causal_conv1d_host_meta`（[683-712](vllm_ascend/ops/gdn_attn_builder.py#L683-L712)）只写 `[:num_spec_decodes]` 前缀，pinned buffer 的 tail 保留双缓冲池里上一轮的脏值（隐患 2）。一旦下游读全长，立刻污染。

**这是解释 MTP=1/2 OK 而 MTP=3 不 OK 的最直接机制。**

### 3.2 假设 B：fast path 内 `spec_q_per_seq` padding 粒度过粗

[vllm_ascend/ops/gdn.py:139](vllm_ascend/ops/gdn.py#L139)：

```python
pad_seqs = pad_tokens // q_per_seq
```

`q_per_seq = num_spec + 1`。MTP=3 时 `q_per_seq = 4`，padding 粒度最粗（4 个 token 一组）。

如果某些场景需要补 1~3 个 token（例如 FIA 加了 dummy request、scheduler 串了 prefill token）：
- MTP=1 (q_per_seq=2)：余 1 个 token 时无法 padding → 但能少补
- MTP=2 (q_per_seq=3)：余 1-2 个 token 时无法 padding
- MTP=3 (q_per_seq=4)：余 1-3 个 token 时无法 padding → 范围最大

无法整除时，tiling 校验 `qsl[last] == cuSeqlen` 失败或被强行截断，可能导致状态错位。

### 3.3 假设 C：bs=16 正好处于 bucket 边界，调度抖动放大效应

MTP=3 实际 batch_size = 16，**恰好等于 `decode_cudagraph_max_bs=16`**，没有余量。一旦：

- 调度器某次迭代加入第 5 个 request（即使是要 decode 一个 token），batch 就会从 16 涨到 20，超过 16 → 走 [1253 行 `_attach_spec_decode_fallback_meta`](vllm_ascend/ops/gdn_attn_builder.py#L1253) 慢路径。
- 慢路径 pinned buffer 有脏值（隐患 2）→ 精度崩。

而 MTP=1 (实际 8) 和 MTP=2 (实际 12) 在 bucket 16 内还有 4-8 的余量，调度抖动不会顶出 fast path。

### 3.4 假设 D：conv1d kernel 的 `spec_q_per_seq=4` 触发未验证 tiling 分支

[vllm_ascend/ops/gdn.py:439](vllm_ascend/ops/gdn.py#L439)：

```python
spec_q_per_seq = int(attn_metadata.spec_state_indices_tensor.size(-1))
```

MTP=3 第一次让 `spec_q_per_seq` 变成 4。`npu_causal_conv1d_custom` 内部某个 tiling 分支可能没在 `q_per_seq=4` 下被测试过，导致写错 conv_state slot。

属于低概率但需要排除的硬件/kernel 层 bug。

---

## 四、根因假设排序

| 排序 | 假设 | 验证成本 | 修复成本 |
|---|---|---|---|
| 🥇 | **假设 A**：`max_cudagraph_capture_size` 把 MTP=3 顶出 fast path → fallback 路径 pinned buffer 脏值 | 极低（打印一行配置） | 低（要么抬高 cap，要么修 fallback pinned buffer 清零） |
| 🥈 | **隐患 1**：fast path 内 `spec_token_indx` / `non_spec_token_indx` 尾部不清零 | 低（加一行 fill 看是否恢复） | 极低（加两行 fill） |
| 🥉 | **假设 C**：bs=16 抖动到 fallback 路径触发隐患 2 | 中（需要日志确认是否真的抖动到 fallback） | 低 |
| 4 | **假设 B**：`q_per_seq=4` padding 粒度过粗 | 中（在 `_pad_conv1d_host_args_to_capture` 里加日志） | 中 |
| 5 | **假设 D**：conv1d kernel tiling on `q_per_seq=4` | 高（需要和 kernel owner 一起 trace） | 高 |
| 6 | **隐患 3**：`extract_hidden_states_proposer.py` 硬编码 num_spec=1 | 低（确认是否在调用链上） | 中 |

---

## 五、排查与验证步骤

### 5.1 第一步：确认 fast path 是否真的进了

在 [vllm_ascend/ops/gdn_attn_builder.py](vllm_ascend/ops/gdn_attn_builder.py) `build()` 开头加日志：

```python
logger.info(
    "GDN build: MTP=%d decode_cudagraph_max_bs=%d "
    "use_full_cuda_graph=%s num_spec_decodes=%d "
    "num_spec_decode_tokens=%d max_cudagraph_cap=%s "
    "fast_path_taken=%s",
    self.num_spec,
    self.decode_cudagraph_max_bs,
    self.use_full_cuda_graph,
    num_spec_decodes,
    num_spec_decode_tokens,
    self.compilation_config.max_cudagraph_capture_size,
    # 在 fast path if 块里设置一个标志变量后再打印
)
```

跑 MTP=3 + 4 reqs，看每次 decode 是不是真的进了 [1152-1201](vllm_ascend/ops/gdn_attn_builder.py#L1152-L1201) 的 fast path。

### 5.2 第二步：enforce_eager 对照

`enforce_eager=True` 跑 MTP=3 + 4 reqs：
- 精度恢复 → 100% 锁定图模式 padding/pinning 脏值问题，可排除算法/kernel 问题
- 仍然有问题 → 算法或 kernel 问题，需要在 eager 模式下逐层对比

### 5.3 第三步：临时修复尝试（验证隐患 1）

在 [vllm_ascend/ops/gdn_attn_builder.py:1186 后](vllm_ascend/ops/gdn_attn_builder.py#L1186) 加：

```python
self.spec_token_indx[spec_token_indx.size(0):].fill_(0)
self.non_spec_token_indx[non_spec_token_indx.size(0):].fill_(0)
```

跑 MTP=3 看精度是否恢复。如果恢复 → 隐患 1 就是元凶。

### 5.4 第四步：逐层 hidden state 对比

在 eager 模式下，逐层 dump：

- conv1d 输入 `mixed_qkv_spec`
- conv1d 输出
- ssm 输入 `q` / `k` / `v`
- ssm 输出
- layer 输出 hidden_states

对比 MTP=2 vs MTP=3 的差异，定位最早出现偏差的层。

### 5.5 第五步：核对 fallback pinned buffer 清零

如果第四步指向 conv1d 阶段，且第一步确认走了 fallback 路径，检查 `_build_spec_causal_conv1d_host_meta`（[683-712](vllm_ascend/ops/gdn_attn_builder.py#L683-L712)）中 `cache_indices_cpu` / `num_accepted_tokens_cpu` 是否被某个下游 consumer 读了全长。可在 `_copy_to_pinned_cpu` 后追加：

```python
# Debug: 清零 tail，看是否恢复精度
cpu_tensor_full = slot.cache_indices_cpu
cpu_tensor_full[num_elements:].fill_(PAD_SLOT_ID)
```

---

## 六、候选修复方案

### 方案 1（最小修复，先验证）：tail 清零

在 [vllm_ascend/ops/gdn_attn_builder.py:1186 后](vllm_ascend/ops/gdn_attn_builder.py#L1186) 加：

```python
self.spec_token_indx[spec_token_indx.size(0):].fill_(0)
self.non_spec_token_indx[non_spec_token_indx.size(0):].fill_(0)
```

### 方案 2（兜底修复）：fallback pinned buffer 整池清零

修改 `_acquire_spec_causal_conv1d_host_slot`（[591-596](vllm_ascend/ops/gdn_attn_builder.py#L591-L596)），acquire 时把整个 slot 清零：

```python
def _acquire_spec_causal_conv1d_host_slot(builder):
    pool = builder._ascend_gdn_spec_causal_conv1d_host_pool
    builder._ascend_gdn_spec_causal_conv1d_host_pool_idx = (
        builder._ascend_gdn_spec_causal_conv1d_host_pool_idx + 1
    ) % len(pool)
    slot = pool[builder._ascend_gdn_spec_causal_conv1d_host_pool_idx]
    slot.cache_indices_cpu.fill_(PAD_SLOT_ID)
    slot.num_accepted_tokens_cpu.fill_(1)
    return slot
```

同样的模式也应用于 `_acquire_causal_conv1d_host_slot`（[547-550](vllm_ascend/ops/gdn_attn_builder.py#L547-L550)）。

### 方案 3（结构性修复）：把"是否需要 tail 清零"做成显式不变量

`spec_token_indx` / `non_spec_token_indx` 的 tail 应该用 sentinel（例如 `PAD_SLOT_ID` 或 0）填充，并通过断言保证下游 consumer 不会越界读。可以封装一个 `_fill_token_indx_buffer(static_buffer, runtime_indices)` helper，集中管理。

### 方案 4（配置层修复）：调整 `max_cudagraph_capture_size`

如果根因是假设 A，最简单的修复是把 `max_cudagraph_capture_size` 抬高到 ≥ `max_num_seqs * (max_MTP + 1)`，确保 MTP=3 也能进 fast path。但这只是 workaround，隐患 2 仍然存在。

---

## 七、附：相关环境与配置

排查时需要确认的配置：

```bash
# 关键环境变量
export VLLM_ASCEND_LOG_LEVEL=DEBUG

# vLLM 配置
--max-num-seqs 4
--spec-num-speculative-tokens 3            # num_speculative_tokens
--compilation-config '{"cudagraph_mode": "FULL", "max_cudagraph_capture_size": ???}'

# 是否启用 enforce_eager 对照
--enforce-eager
```

需要重点确认的运行时参数：

- `max_num_seqs`（scheduler）
- `compilation_config.cudagraph_mode`（是否 FULL）
- `compilation_config.max_cudagraph_capture_size`（关键！）
- `speculative_config.method`（eagle / dflash / ...）
- `speculative_config.num_speculative_tokens`

---

## 八、附：核心代码引用一览

### 8.1 fast path 触发条件

[vllm_ascend/ops/gdn_attn_builder.py:1152-1158](vllm_ascend/ops/gdn_attn_builder.py#L1152-L1158)：

```python
if (
    self.use_full_cuda_graph
    and num_prefills == 0
    and num_decodes == 0
    and num_spec_decodes <= self.decode_cudagraph_max_bs
    and num_spec_decode_tokens <= self.decode_cudagraph_max_bs
):
```

### 8.2 `decode_cudagraph_max_bs` 计算

[vllm/v1/attention/backends/gdn_attn.py:98-105](vllm/v1/attention/backends/gdn_attn.py#L98-L105)：

```python
self.decode_cudagraph_max_bs = (
    self.vllm_config.scheduler_config.max_num_seqs * (self.num_spec + 1)
)
if self.compilation_config.max_cudagraph_capture_size is not None:
    self.decode_cudagraph_max_bs = min(
        self.decode_cudagraph_max_bs,
        self.compilation_config.max_cudagraph_capture_size,
    )
```

### 8.3 pinned buffer 容量

[vllm_ascend/ops/gdn_attn_builder.py:553-575](vllm_ascend/ops/gdn_attn_builder.py#L553-L575)：

```python
def _allocate_spec_causal_conv1d_host_slot(builder, device):
    max_num_seqs = builder.vllm_config.scheduler_config.max_num_seqs
    spec_cfg = builder.vllm_config.speculative_config
    num_speculative_tokens = spec_cfg.num_speculative_tokens if spec_cfg else 0
    decode_cudagraph_max_bs = getattr(builder, "decode_cudagraph_max_bs", max_num_seqs)
    max_elements = decode_cudagraph_max_bs * (num_speculative_tokens + 1)
    ...
```

### 8.4 conv1d host args padding

[vllm_ascend/ops/gdn.py:106-148](vllm_ascend/ops/gdn.py#L106-L148)：

```python
def _pad_conv1d_host_args_to_capture(
    qsl_host, cidx_host, num_accepted_host,
    cap_x_dim0, q_per_seq, with_num_accepted,
):
    ...
    pad_tokens = cap_x_dim0 - runtime_qsl_last
    if pad_tokens <= 0 or q_per_seq <= 0:
        return qsl_host, cidx_host, num_accepted_host
    pad_seqs = pad_tokens // q_per_seq
    ...
```
