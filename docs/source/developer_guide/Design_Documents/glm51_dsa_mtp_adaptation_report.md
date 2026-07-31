# GLM-5.1 DSA Sparse Offload + MTP 适配技术报告

## 1. 概述

| 项目 | 内容 |
|---|---|
| 基线分支 | `releases/v0.23.0` |
| 适配分支 | `releases/v0.23.0-glm` |
| 目标模型 | GLM-5/5.1（架构 `GlmMoeDsaForCausalLM`，model_type `glm_moe_dsa`） |
| 目标场景 | DSA sparse KV offload + 图模式（FULL_DECODE_ONLY）+ DP1EP1 + MTP |
| 算子 ABI | `LightningIndexerDecodeUpdate`（12 参数）和 `KvcacheScatterCopy`（9 参数）入参出参保持不变 |

## 2. 适配前状态

基线 `releases/v0.23.0-glm`（commit `6edf10ec7`）存在以下问题：

### 2.1 RoPE 精度问题

GLM-5.1 使用 interleaved（GPT-J 风格）RoPE（`is_rope_neox_style=False`），但非 Triton 路径无条件使用 `npu_rotary_mul`（标准非交错式），导致 Indexer Q/K 旋转位置编码错误。

commit `f9bd0ef27` 尝试修复但仅替换算子名，未改变调用约定（传入 split 后的 rope 部分 + 4D cos/sin），`npu_interleave_rope` 期望完整 head_dim + 2D cos/sin，行为不可预测。

commit `758421e50` 引入 `_restore_npu_interleave_rope_layout` 后处理和统一方法 `_apply_indexer_rope`，但 cos/sin 仍为 4D，调用约定仍不一致。

### 2.2 MTP 功能被 fail-closed 拒绝

commit `6edf10ec7` 直接拒绝所有 `speculative_config`，无法联合使用 DSA sparse offload 和 MTP。

### 2.3 DSA + MTP 联合执行崩溃

MTP draft model 的 attention metadata 构建跳过 Indexer 层（`llm_base_proposer.py:330-336` 显式排除 `all_indexer_layer_names`），DSA sparse 路径访问 `forward_context.attn_metadata[indexer_layer_name]` 时 KeyError → RuntimeError。

## 3. 适配改动

### 3.1 改动清单

| # | 文件 | 修改类型 | 说明 |
|---|---|---|---|
| 1 | `vllm_ascend/attention/sfa_v1.py` | 新增方法 + 条件 guard | DSA 图模式执行路径 + MTP draft 兼容 |
| 2 | `vllm_ascend/dsa_sparse/dsa_config.py` | 恢复 + 重构 | 恢复 MTP 支持 + 图模式配置 + 配置校验 |
| 3 | `vllm_ascend/patch/dsa_sparse/patch_scheduler.py` | 新增 | MTP block 边界裁剪 + scheduler 输出 re-bind |
| 4 | `vllm_ascend/platform.py` | 条件 guard | 图模式时不禁用 `enforce_eager` |
| 5 | `vllm_ascend/worker/model_runner_v1.py` | 新增 | DSA trace + 图模式 metadata 构建 |
| 6 | `examples/glm51_dsa_sparse_mtp.sh` | 重写 | 环境变量驱动的启动脚本 |

### 3.2 核心改动详解

#### 3.2.1 DSA Indexer metadata guard（`sfa_v1.py:2107-2141`）

**问题：** MTP draft model 经过编译装饰器后 `is_draft_model` 标志丢失。

**方案：** 在进入 DSA indexer 解析前，检查 `forward_context.attn_metadata` 中是否实际存在 Indexer 层的 metadata：

```python
if self._dsa_split_indexer_cache_enabled() and not _EXTRA_CTX.is_draft_model:
    forward_context = get_forward_context()
    all_metadata = forward_context.attn_metadata
    indexer_layer_name = self.indexer_k_cache_layer_name
    if (isinstance(all_metadata, dict)
            and indexer_layer_name is not None
            and indexer_layer_name in all_metadata):
        # 主模型路径：解析 Indexer metadata
        ...
        # MTP draft 路径：metadata 不存在 → dsa_mgr 保持 None → 走原生 SFA
```

#### 3.2.2 Indexer KV cache 写入 guard（`sfa_v1.py:2395`）

**问题：** DSA split cache 模式下 `kv_cache` 只有 2 个元素 `[nope, rope]`，`else` 分支访问 `kv_cache[2]` → IndexError。

**方案：** `else:` → `elif not self._dsa_split_indexer_cache_enabled():`，DSA split cache 模式下跳过 Indexer KV cache 写入（MTP draft 不需要独立 Indexer cache）。

#### 3.2.3 MTP draft dense SFA fallback（`sfa_v1.py:2501-2510`）

**问题：** MTP draft step 0（`skip_topk=False`）调用 `indexer_select_post_process` → `device_op.py:620` 访问 `kv_cache[2]` → IndexError。

**方案：** 当 `dsa_mgr is None` 且 DSA split cache 启用时，构造全零 `topk_indices`，走 dense SFA（全量 attention，不做 top-K 稀疏选择）：

```python
elif dsa_mgr is None and self._dsa_split_indexer_cache_enabled():
    topk_indices = torch.zeros(
        (topk_num_tokens, 1, 1),
        dtype=torch.int32,
        device=ql_nope.device,
    )
```

#### 3.2.4 DSA 图模式执行路径（`sfa_v1.py:1917-1997`）

**新增方法：** `_execute_dsa_offload_graph` — 图 capture/replay 期间的 DSA offload 执行路径。

图 gate 保证每行为单 token decode，MTP round 循环塌缩为单轮。所有行选择使用预构建固定地址张量，避免 `.cpu()` / `.item()` / `np.flatnonzero` 破坏图 capture/replay。

#### 3.2.5 MTP block 边界裁剪（`patch_scheduler.py:122-167`）

**新增方法：** `_trim_dsa_mtp_drafts_at_block_boundaries` — 防止未验证 MTP token 完成 offloaded MLA block。

DRAM full-block dump 发布后不能按 draft rejection 回滚。scheduler 在 speculative step 进入完整 MLA block 边界前裁剪 draft，保证 token 恰好完成 block 时本 step draft 数为 0。

#### 3.2.6 恢复 MTP 支持（`dsa_config.py:510-518`）

将 fail-closed 拒绝改回安全文档：

```python
# MTP is supported in eager mode.  The scheduler patch trims draft tokens
# at MLA block boundaries so an unverified draft can never complete
# (and thus trigger a DRAM dump of) an offloaded MLA block.  In graph mode,
# the graph gate rejects multi-token decode, so MTP steps fall back to
# eager automatically.
```

#### 3.2.7 图模式配置支持（`platform.py:470-483` + `dsa_config.py:486-554`）

- 当 `enable_row_mode_decode_graph=true` 时，不强制 `enforce_eager=True`
- 允许原生 FULL graph family capture/replay 单 token decode
- 拒绝 `npugraph_ex=true`（会在 KV-cache split metadata 初始化前失败）

## 4. 执行路径矩阵

### 4.1 主模型 forward

| 步骤 | 变量 | 值 | 走向 |
|---|---|---|---|
| DSA 分支 | `_dsa_split_indexer_cache_enabled()` | True | 进入 |
| metadata guard | `indexer_layer_name in all_metadata` | True | 解析 Indexer metadata |
| `dsa_mgr` | `get_dsa_worker_manager()` | not None | DSA active |
| Indexer KV 写入 | `indexer_k_cache is not None` | True | 写入独立 Indexer cache |
| attention 路径 | `dsa_row_mode_active` | True | LIDU→KSC→SFA-Offload |
| 图 capture | `_EXTRA_CTX.capturing` | True/False | graph/eager 两路径 |

### 4.2 MTP draft step 0（`skip_topk=False`）

| 步骤 | 变量 | 值 | 走向 |
|---|---|---|---|
| DSA 分支 | `is_draft_model` | None（编译后丢失） | 进入（`not None = True`） |
| metadata guard | `indexer_layer_name in all_metadata` | False | 跳过，`dsa_mgr=None` |
| Indexer KV 写入 | `indexer_k_cache is None` | True | `elif not _dsa_split...` → False → 跳过 |
| topk 选择 | `dsa_mgr is None and _dsa_split...` | True | 全零 topk_indices → dense SFA |
| `attention_finished` | `dsa_mgr is None` | True | 跳过 |

### 4.3 MTP draft step 1+（`skip_topk=True`）

| 步骤 | 变量 | 值 | 走向 |
|---|---|---|---|
| DSA 分支 | 同 step 0 | — | `dsa_mgr=None` |
| topk 选择 | `self.skip_topk` | True | `_get_indexcache_topk_indices`（读取共享 buffer） |
| SFA attention | — | — | 使用 buffer 中的 topk_indices |

### 4.4 非 DSA 部署

| 步骤 | 变量 | 值 | 走向 |
|---|---|---|---|
| DSA 分支 | `_dsa_split_indexer_cache_enabled()` | False | 跳过 |
| 所有 DSA 路径 | — | — | 不受影响 |

## 5. 算子 ABI 验证

| 算子 | 参数数量 | 参数顺序 | 原地写入标记 | csrc/ 修改 |
|---|---|---|---|---|
| `npu_lightning_indexer_decode_update_out` | 12 | 不变 | `(a!)(b!)(c!)(d!)(e!)` | 零修改 |
| `npu_kvcache_scatter_copy` | 9 | 不变 | `(a!)(b!)` | 零修改 |

## 6. 框架适配原则合规

| 原则 | 合规 | 说明 |
|---|---|---|
| 不改原生生态代码 | ✅ | 所有修改在 `vllm_ascend/` 内，不修改上游 vLLM |
| 改动最小 | ✅ | 6 个文件，+1157/-138 行；所有修改为条件分支或新增方法 |
| patch 模式 | ✅ | DSA 通过 `additional_config` 驱动，未启用时不执行 |
| 幂等安全 | ✅ | 所有 guard 为确定性条件分支 |

## 7. 启动方式

```bash
# DSA + eager + MTP
DSA_ENABLED=true DSA_GRAPH_ENABLED=false MTP_ENABLED=true \
  bash examples/glm51_dsa_sparse_mtp.sh

# DSA + 图模式 + MTP
DSA_ENABLED=true DSA_GRAPH_ENABLED=true MTP_ENABLED=true \
  bash examples/glm51_dsa_sparse_mtp.sh

# DSA + 图模式（无 MTP）
DSA_ENABLED=true DSA_GRAPH_ENABLED=true MTP_ENABLED=false \
  bash examples/glm51_dsa_sparse_mtp.sh
```

## 8. 已知限制

1. **MTP draft 走 dense SFA**：draft model 不做 top-K 稀疏选择，使用全量 attention，性能不如主模型的 DSA sparse 路径
2. **GLM 系列 draft 强制 eager**：`llm_base_proposer.py:225-234` 强制 `use_cuda_graph=False`，draft model 不经过 `ACLGraphWrapper`
3. **图模式仅 capture 单 token decode**：图 gate 拒绝多 token decode（`non_single_token_decode`），MTP 多 token 步骤自动回退 eager
4. **MTP block 边界裁剪**：scheduler 在 MLA block 边界裁剪 draft，可能减少 draft token 数量
5. **真机验证待完成**：DSA 自定义算子（LIDU/KSC/SFA-Offload/FullBlockDump）的编译、精度和性能仍需在 A2/A3 真机验收
