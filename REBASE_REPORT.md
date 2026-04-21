# Rebase 报告：dev-dycp-rebase-0.18 onto releases/v0.18.0

## 概述

- **源分支**：`dev-dycp-rebase-0.18`（31 个 dycp 相关 commit）
- **目标 base**：`releases/v0.18.0`（286 个 commit）
- **共同祖先**：`a78a00e0` ([Doc][ReleaseNote] Add release notes for v0.16.0rc1)
- **备份分支**：`dev-dycp-rebase-0.18-backup`
- **涉及文件**：13 个文件，1591 行增加，600 行删除

## 冲突文件概览

| 文件 | 冲突次数 | 风险等级 |
|------|---------|---------|
| `vllm_ascend/worker/model_runner_v1.py` | 7 次（跨 6 个 commit） | 高 |
| `vllm_ascend/attention/context_parallel/mla_cp.py` | 5 次（跨 3 个 commit） | 中 |
| `csrc/utils/inc/kernel/moe_distribute_base.h` | 13 个冲突块（1 个 commit） | 中 |
| `vllm_ascend/attention/mla_v1.py` | 1 次 | 低 |

以下无冲突的 dycp 文件顺利 rebase：
- `vllm_ascend/ascend_forward_context.py`
- `vllm_ascend/attention/context_parallel/common_cp.py`
- `vllm_ascend/attention/utils.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`
- `vllm_ascend/worker/npu_input_batch.py`
- `vllm_ascend/worker/pcp_utils.py`
- `vllm_ascend/worker/worker.py`

---

## 逐 Commit 冲突详情与解决方案

### 1. `2f352e8a` — [feat] support pcp dynamic

**冲突文件**：`model_runner_v1.py`（3 处冲突）

#### 冲突 1.1：import 语句（第 39 行附近）

| 侧 | 内容 |
|----|------|
| releases/v0.18.0 | `from vllm.distributed.parallel_state import get_dcp_group, get_dp_group, get_pcp_group, get_pp_group, get_tp_group` <br> `from vllm.forward_context import BatchDescriptor, ForwardContext, get_forward_context` |
| dycp 分支 | 增加了 `get_dycp_group`，但没有 import `ForwardContext` |

**解决方案**：两边合并 — 保留 v0.18.0 的 `ForwardContext` import，同时添加 dycp 的 `get_dycp_group`。

```python
from vllm.distributed.parallel_state import get_dcp_group, get_dp_group, get_pcp_group, get_pp_group, get_tp_group, get_dycp_group
from vllm.forward_context import BatchDescriptor, ForwardContext, get_forward_context
```

**理由**：`ForwardContext` 在 v0.18.0 中被 `_build_attn_metadata` 等函数使用（类型注解），必须保留。`get_dycp_group` 是 dycp 特性所需。

#### 冲突 1.2：CUDA Graph 条件判断（第 1303 行附近）

| 侧 | 内容 |
|----|------|
| releases/v0.18.0 | `and self.pcp_size * self.dcp_size == 1` |
| dycp 分支 | `and not self.use_prefill_cp()  # TODO(lxs): fix this` |

**解决方案**：采用 dycp 分支的 `use_prefill_cp()` 版本。

**理由**：
- v0.18.0 的 `self.pcp_size * self.dcp_size == 1` 仅检查静态 pcp/dcp 大小
- dycp 的 `use_prefill_cp` 是一个 `@property`，返回 `self.pcp_size > 1 or self.prefill_dycp_size > 1`，能同时处理静态 CP 和动态 CP 的情况
- 对于非 dycp 场景（`prefill_dycp_size <= 1`），行为等价

#### 冲突 1.3：Forward Pass 结构（第 1388 行附近）— **最复杂的冲突**

| 侧 | 结构 |
|----|------|
| releases/v0.18.0 | 单一 `with` 块，使用 `defer_finalize` 参数，`self.pcp_size == 1` |
| dycp 分支 | `if vllm_version_is("0.16.0")` 分支判断 + 两个 `with` 块，使用 `use_prefill_cp()` |

**解决方案**：采用 v0.18.0 的单一 `with` 块结构 + `defer_finalize` API，将 `self.pcp_size == 1` 替换为 `not self.use_prefill_cp()`。

```python
with (
    record_function_or_nullcontext("forward"),
    set_ascend_forward_context(
        ...,
        max_tokens_across_pcp=0 if not self.use_prefill_cp() else self.pcp_manager.max_num_tokens_across_pcp,
        ...
    ),
    self.maybe_get_kv_connector_output(
        scheduler_output,
        **({"defer_finalize": not clear_kv_metadata}),
    ) as kv_connector_output,
):
    hidden_states = self._model_forward(...)
```

**理由**：
- 既然 rebase 到 v0.18.0，不再需要 `vllm_version_is("0.16.0")` 版本兼容分支
- v0.18.0 的 `maybe_get_kv_connector_output` API 使用 `defer_finalize` 参数，而非旧版的 `clear_metadata`
- dycp 的 `use_prefill_cp` 逻辑需要保留

---

### 2. `7d029243` — [fix] fix pcp switch

**冲突文件**：`model_runner_v1.py`（1 处冲突，同 1.3 区域）

**冲突本质**：这个 commit 把 `self.use_prefill_cp()` 修复为 `self.use_prefill_cp`（去掉括号，因为是 `@property`）。但它仍然携带了 `vllm_version_is` 版本分支结构。

**解决方案**：同上一个冲突的结构，但应用 property 修复：

```python
max_tokens_across_pcp=0 if not self.use_prefill_cp else ...
```

**理由**：`use_prefill_cp` 定义为 `@property`，调用时不需要括号。这是一个真实的 bug fix。

---

### 3. `e27e6ec8` — [debug] bugfix

**冲突文件**：`model_runner_v1.py`（1 处冲突）

**冲突位置**：`_build_attn_metadata` 函数签名（第 2333 行附近）

| 侧 | 新增参数 |
|----|---------|
| releases/v0.18.0 | `profile_seq_lens: int \| None = None` |
| dycp 分支 | `num_dycp_reqs: int = 0` |

**解决方案**：两个参数都保留，按顺序排列。

```python
profile_seq_lens: int | None = None,
num_dycp_reqs: int = 0,
```

**理由**：两个参数互不相关，分别服务于不同功能（性能分析 vs 动态 CP）。

---

### 4. `b761b780` — [feat] support acl graph

**冲突文件**：3 个（`moe_distribute_base.h`、`mla_cp.py`、`model_runner_v1.py`）

#### 冲突 4.1：`csrc/utils/inc/kernel/moe_distribute_base.h`（13 个冲突块）

**冲突类型**：add/add — 两个分支独立创建了这个文件。

| 侧 | 行数 | 差异 |
|----|------|------|
| releases/v0.18.0 | 287 行 | 纯结构体定义，无注释 |
| dycp 分支 | 364 行 | 相同结构体 + 中文注释 + 额外功能代码 |

dycp 分支独有的新增代码：
- `#include "kernel_operator.h"`
- `constexpr uint32_t TIME_CYCLE = 50;`
- `struct CombinedCapability` + `enum class DataplaneMode`
- `__aicore__ inline DataplaneMode GetDataplaneMode(...)`
- `__aicore__ inline int64_t GetCurrentTimestampUs()`
- `__aicore__ inline void RecordRankCommDuration(...)`

**解决方案**：完整采用 dycp 分支版本（`git checkout --theirs`）。

**理由**：dycp 版本是 v0.18.0 版本的超集 — 包含所有相同的结构体定义，额外添加了中文字段注释和 ACL graph 所需的辅助函数。

#### 冲突 4.2：`mla_cp.py`（第 869 行附近）

| 侧 | 内容 |
|----|------|
| releases/v0.18.0 | `if _EXTRA_CTX.capturing:` |
| dycp 分支 | 新增 `graph_key = (num_tokens, num_dycp_reqs) if num_dycp_reqs > 0 else num_tokens` + `if forward_context.capturing:` |

**解决方案**：保留 dycp 新增的 `graph_key` 行，但将 `forward_context.capturing` 改为 `_EXTRA_CTX.capturing`。

```python
graph_key = (num_tokens, num_dycp_reqs) if num_dycp_reqs > 0 else num_tokens
if _EXTRA_CTX.capturing:
```

**理由**：
- `graph_key` 是 dycp 特性核心逻辑，用于在 ACL graph 中区分不同 dycp 请求数的缓存
- v0.18.0 使用 `_EXTRA_CTX` 而非 `forward_context` 来访问 capturing 状态

#### 冲突 4.3：`model_runner_v1.py` — `dispatch_cudagraph` 函数（第 2031 行附近）

| 侧 | 内容 |
|----|------|
| releases/v0.18.0 | `self.cudagraph_dispatcher.dispatch(... valid_modes=..., invalid_modes=...)` |
| dycp 分支 | 增加 `num_dycp_reqs=num_dycp_reqs` 参数 + `vllm_version_is` 分支 |

**解决方案**：使用 v0.18.0 的 `valid_modes`/`invalid_modes` API + 添加 `num_dycp_reqs`。

```python
return self.cudagraph_dispatcher.dispatch(
    num_tokens=num_tokens,
    has_lora=has_lora,
    uniform_decode=uniform_decode,
    valid_modes=valid_modes,
    invalid_modes={CUDAGraphMode.FULL} if disable_full else None,
    num_active_loras=num_active_loras,
    num_dycp_reqs=num_dycp_reqs,
)
```

#### 冲突 4.4：`model_runner_v1.py` — DP re-dispatch（第 2083 行附近）

同 4.3 模式。v0.18.0 的 `valid_modes` API + dycp 的 `num_dycp_reqs` 参数。

#### 冲突 4.5：`model_runner_v1.py` — forward pass + `set_ascend_forward_context`

同冲突 1.3 模式。增加了 `num_cp_reqs=scheduler_output.num_cp_request` 参数。

---

### 5. `5384f85e` — [bugfix] PD bug & graph bug

**冲突文件**：3 个（`mla_cp.py`、`mla_v1.py`、`model_runner_v1.py`）

#### 冲突 5.1：`mla_cp.py` — helper 函数注入（第 375 行附近）

| 侧 | 内容 |
|----|------|
| releases/v0.18.0 | 直接 `if _EXTRA_CTX.is_draft_model:` |
| dycp 分支 | 在前面插入了 4 个 helper 函数 `_seq_len`、`_to_list`、`_align_to_target_len`、`_align_seq_like` |

**解决方案**：保留所有 helper 函数（dycp 的 PD 场景需要用于对齐不同长度的序列元数据），将 `forward_context.is_draft_model` 改为 `_EXTRA_CTX.is_draft_model`。

**理由**：这些 helper 用于动态 CP 场景下 prefill/decode 分离时，不同 rank 之间序列长度可能不一致，需要对齐。

#### 冲突 5.2 / 5.3：`mla_cp.py` — `_EXTRA_CTX` vs `forward_context`（第 1050、1061 行附近）

统一改为 `_EXTRA_CTX.is_draft_model` 和 `_EXTRA_CTX.capturing`。

#### 冲突 5.4：`mla_v1.py`（第 1642 行附近）

| 侧 | 内容 |
|----|------|
| releases/v0.18.0 | 空行（不需要手动获取 forward_context） |
| dycp 分支 | `forward_context = get_forward_context()` |

**解决方案**：使用 v0.18.0 版本（空行）。

**理由**：v0.18.0 的代码路径不需要在此处显式获取 `forward_context`，后续代码通过其他方式访问。

#### 冲突 5.5：`model_runner_v1.py` — seq_lens 清零（第 809 行附近）

| 侧 | 内容 |
|----|------|
| releases/v0.18.0 | `self.seq_lens.cpu[num_reqs:].fill_(0)` |
| dycp 分支 | `self.seq_lens.np[num_reqs:] = 0`（带注释解释原因） |

**解决方案**：采用 dycp 版本（`.np` + 注释）。

**理由**：
- 功能等价，都是清零 seq_lens 尾部
- dycp 版本的注释解释了为什么需要清零：full-graph 模式下 metadata builder 会消费 `seq_lens_cpu[:num_reqs_padded]`，如果尾部残留旧值，padded request 会继承错误的长度，导致 attention 输入不正确
- `.np` 访问更直接

#### 冲突 5.6：`model_runner_v1.py` — forward pass（第 1496 行附近）

dycp 分支在 `with` 块内添加了 `kv_connector_output.finished_sending/recving` 的 dycp 处理逻辑。但这些逻辑在后续的 post-process 块中已经存在（由之前已成功 rebase 的 commit 添加）。

**解决方案**：使用 HEAD（v0.18.0 结构），避免重复逻辑。

---

### 6. `0c620703` — [feat] support chunkprefill

**冲突文件**：`mla_cp.py`（1 处冲突）

**冲突位置**：import 语句（第 14 行）

| 侧 | 内容 |
|----|------|
| releases/v0.18.0 | `from vllm.utils.math_utils import cdiv` |
| dycp 分支 | 增加 `ForwardContext, get_forward_context` import 和 `round_down` |

**解决方案**：全部保留。

```python
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils.math_utils import cdiv, round_down
```

**理由**：`round_down` 在 `split_attn_metadata` 函数中实际使用（`max_context_chunk = round_down(max_context_chunk, block_size)`）。`get_forward_context` 在文件中仍有调用点。

---

### 7. `b2aea82c` — bugfix for domain/mooncake/modelrunner

**冲突文件**：`model_runner_v1.py`（1 处冲突，同 forward pass 区域）

同冲突 5.6 模式。dycp 分支内含 `vllm_version_is` 版本分支和在 with 块内的 kv_connector 处理（已注释掉）。

**解决方案**：使用 HEAD（v0.18.0 结构）。

---

### 8. `d006ec9a` — bugfix for modelrunner

**冲突文件**：`model_runner_v1.py`（1 处冲突，同 forward pass 区域）

同冲突 7 模式。

**解决方案**：使用 HEAD（v0.18.0 结构）。

---

## 系统性修复总结

### 修复 1：消除 `vllm_version_is("0.16.0")` 版本分支

dycp 原始代码中广泛使用了 `if vllm_version_is("0.16.0"): ... else: ...` 来同时兼容 v0.16.0 和 v0.18.0 的 API。rebase 后这些分支全部消除，统一使用 v0.18.0 的 API：

| v0.16.0 API | v0.18.0 API |
|-------------|-------------|
| `maybe_get_kv_connector_output(scheduler_output)` | `maybe_get_kv_connector_output(scheduler_output, **({"defer_finalize": not clear_kv_metadata}))` |
| `dispatch_cudagraph(..., disable_full=...)` | `dispatch_cudagraph(..., valid_modes=..., invalid_modes=...)` |

### 修复 2：统一 Context 访问模式

| dycp 原始代码 | v0.18.0 适配后 |
|--------------|---------------|
| `forward_context.is_draft_model` | `_EXTRA_CTX.is_draft_model` |
| `forward_context.capturing` | `_EXTRA_CTX.capturing` |

### 修复 3：保留 `use_prefill_cp` property 修复

确保所有调用点使用 `self.use_prefill_cp`（无括号），而非 `self.use_prefill_cp()`。

---

## 后续建议

1. **功能验证**：建议在 NPU 环境下运行 dycp 相关测试，特别关注：
   - PD 分离场景（prefill/decode 拆分）
   - ACL graph capture/replay
   - mooncake connector 的 kv transfer
   - chunkprefill 功能

2. **`_get_req_cp_size` 逻辑位置**：post-process 块中的 `kv_connector_output.finished_sending/recving` 处理逻辑是从 forward context 内移出来的，需要确认时序上是否正确（即在 forward 完成后处理是否等价于在 forward 过程中处理）。

3. **`dispatch_cudagraph` 的 `num_dycp_reqs` 参数**：需要确认 v0.18.0 的 `cudagraph_dispatcher.dispatch` 方法是否已支持 `num_dycp_reqs` 关键字参数，否则运行时会报 `TypeError`。

4. **回退方案**：如需回退，执行：
   ```bash
   git checkout dev-dycp-rebase-0.18-backup
   git branch -D dev-dycp-rebase-0.18
   git branch -m dev-dycp-rebase-0.18
   ```
