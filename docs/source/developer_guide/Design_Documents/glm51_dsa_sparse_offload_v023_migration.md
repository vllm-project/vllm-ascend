# GLM-5.1 DSA 稀疏 KV Offload 升级至 v0.23.0 改动说明

## 1. 文档信息

| 项目 | 内容 |
|---|---|
| 目标框架 | `long1016033679/vllm-ascend:releases/v0.23.0` |
| 目标基线提交 | `98c40c6602fa2005f50676451f05f535b7ad142e` |
| 迁移来源 | `long1016033679/vLLM-ascend-DSA:vllm-ascend-v0.19.1rc1-gs-glm` |
| 目标模型 | GLM-5/GLM-5.1，架构名 `GlmMoeDsaForCausalLM` |
| 推理范围 | decode-only sparse KV offload、eager、DP+TP；不含 speculative/MTP |
| 目标设备 | 昇腾 A2、A3、A5（Ascend 950） |
| 文档日期 | 2026-07-29 |

本文说明迁移后的设计、算子 ABI、代码改动、部署方式、已验证内容和真机验收要求。

## 2. 交付结论

本次改造以 v0.23.0 为完整基线，没有回移旧版本模型实现。迁移内容包括：

1. 保留 `LightningIndexerDecodeUpdate` 和 `KvcacheScatterCopy` 的 ACL 算子名称。
2. 保留两个算子的 PyTorch 方法名、输入参数顺序、输出参数和原地修改语义。
3. A2/A3 使用迁移后的融合自定义算子。
4. A5 在相同 PyTorch 方法名和 tensor ABI 下注册 eager PrivateUse1 组合实现。
5. 将 Indexer 和 MLA KV cache 拆分为独立 cache group 和独立 block pool。
6. 保留 Indexer 全量 HBM cache；MLA 使用 HBM resident window 和 DRAM full-block arena。
7. 将调度状态机、KV cache planning、`NPUInputBatch` sidecar、模型输入整理和 attention 接入点适配到 v0.23.0。
8. 对 speculative/MTP 配置在启动阶段 fail-closed，避免未验收的多 token cache 路径产生错误输出。
9. DP rank 和 TP worker 各自持有请求状态、resident pool、DRAM arena 和算子 backend，不共享可变 KV ownership。
10. 对不在本次支持范围内的组合在启动阶段失败，不进行静默降级。

当前代码已通过无依赖迁移契约测试、Python 编译检查、Shell 语法检查和 `git diff --check`。当前开发环境不包含 CANN、`torch_npu` 和昇腾设备，因此 A2/A3 自定义算子编译、A5 eager 数据面、端到端精度与性能仍必须按第 12 节在目标服务器验收。

## 3. 支持边界

| 能力 | 状态 | 说明 |
|---|---|---|
| GLM-5/GLM-5.1 | 支持 | 仅接受 `GlmMoeDsaForCausalLM` |
| Decode-only sparse offload | 支持 | Prefill 使用原生 dense 路径；完整 MLA block 在层内写入后 dump |
| Eager | 支持且强制 | 自动设置 `model_config.enforce_eager=True` |
| DP+TP | 支持 | DP/TP 大小可以按服务器拓扑配置 |
| Speculative/MTP | 不支持 | 来源 DSA 分支未集成该 cache 语义；配置后启动失败 |
| A2/A3 | 支持 | LIDU、KSC、SFA-Offload、full-block dump 使用融合自定义算子 |
| A5 | 功能适配 | LIDU/KSC 保持同名同参；使用 eager tensor 组合实现，尚需真机性能优化 |
| BF16/FP16 cache | 支持 | 保持来源分支算子 ABI |
| FP8/C8 sparse cache | 不支持 | 启动时拒绝 |
| ACL Graph/TorchAir graph | 不支持 | 本次范围为 eager |
| PP、DCP、PCP | 不支持 | token domain/cache ownership 尚未定义 |
| Prefix caching | 不支持 | 与 DRAM block hash/refcount ownership 冲突 |
| Chunked prefill | 不支持 | `max_num_batched_tokens` 必须覆盖完整 prompt |
| Async scheduling | 不支持 | dump 发布和 scheduler commit 需要同步步进 |
| 通用 KV connector/offloader | 不支持共存 | 避免双重驱逐和 block ownership 冲突 |

## 4. 数据面设计

迁移后保留三类独立存储平面：

| 平面 | 所在介质 | 内容 | 生命周期 |
|---|---|---|---|
| Indexer dense cache | HBM | GLM Indexer 的完整 K cache | 跟随 vLLM request/KV block 生命周期 |
| MLA resident cache | HBM | 每请求固定预算的 resident window 和当前 tail | 由 DSA scheduler sidecar 与独立 block pool 管理 |
| MLA hot store | NPU 可寻址 DRAM | 已完成的 MLA NOPE/ROPE full block | worker-local 固定容量 arena，按 block hash/refcount 管理 |

每个 attention layer 的 decode 顺序如下：

1. 使用 dense Indexer cache 执行 `LightningIndexerDecodeUpdate`。
2. LIDU 在 device 上更新 resident map，并产生 top-K、miss prefix、miss count 和 tail 信息。
3. `KvcacheScatterCopy` 仅把 miss prefix 从 DRAM 搬到 MLA resident HBM slot。
4. `SparseFlashAttentionForOffload` 在 top-2048 resident token 和 dense tail 上计算注意力。
5. 当本层新形成完整 MLA block 时，`KvCacheFullBlockDump` 在同一 NPU stream 上把该 block 写入 DRAM。

热路径不读取 device `miss_count` 到 Host，也不调用 `.item()` 决定 copy 数量。LIDU、KSC 和 SFA-Offload 使用 device tensor 直接串联。

### 4.1 Prefill 与 decode

- Prefill 保留 v0.23.0 原生 dense attention 行为。
- 最后一个非分块 prefill step 会为 prompt 中的每个完整 MLA block 生成 dump 计划。
- 在稀疏激活阈值之前，请求处于 `DENSE_DECODE`。
- 首次达到稀疏条件时进入 `ENTER_SPARSE_DECODE`，建立 resident row。
- 后续进入 `SPARSE_DECODE`，执行 LIDU → KSC → SFA-Offload。
- 当前未完成的 tail 保留在 MLA resident HBM 中；只有完整 block 才发布到 DRAM。

### 4.2 状态机

| 状态 | 含义 | 可执行动作 |
|---|---|---|
| `PREFILL` | 完整 prompt prefill | native attention；dump 已完成 full block |
| `DENSE_DECODE` | context 尚未进入稀疏窗口 | native dense selection/attention |
| `ENTER_SPARSE_DECODE` | 首次建立 resident row | 初始化 resident map；按预算物化 token |
| `SPARSE_DECODE` | 稳态稀疏 decode | LIDU、miss-only KSC、SFA-Offload |

请求被 preempt、结束或释放时，两个 block pool、resident row、DRAM logical block 和 hash/refcount 会在对应的 v0.23.0 生命周期钩子中一起更新。

## 5. 算子兼容性

### 5.1 ACL 算子名称

以下名称保持不变：

| 算子 | ACL 名称 | PyTorch 方法名 |
|---|---|---|
| Lightning Indexer decode/update | `LightningIndexerDecodeUpdate` | `_C_ascend::npu_lightning_indexer_decode_update_out` |
| KV cache scatter copy | `KvcacheScatterCopy` | `_C_ascend::npu_kvcache_scatter_copy` |

### 5.2 `LightningIndexerDecodeUpdate` ABI

```text
npu_lightning_indexer_decode_update_out(
    Tensor query,
    Tensor key,
    Tensor weights,
    Tensor req_pool_entries,
    Tensor(a!) cache_slots,
    Tensor row_modes,
    Tensor actual_seq_lengths_key,
    Tensor block_table,
    Tensor(b!) topk_index_out,
    Tensor(c!) topk_slots_out,
    Tensor(d!) miss_count_out,
    Tensor(e!) tail_info_out
) -> ()
```

方法无返回 tensor；`cache_slots`、`topk_index_out`、`topk_slots_out`、`miss_count_out` 和 `tail_info_out` 保持原地写入语义。

### 5.3 `KvcacheScatterCopy` ABI

```text
npu_kvcache_scatter_copy(
    Tensor(a!) hbm_k_rope,
    Tensor(b!) hbm_kv_cache,
    Tensor dram_k_rope,
    Tensor dram_kv_cache,
    Tensor hbm_block_table,
    Tensor dram_block_table,
    Tensor src_token_ids,
    Tensor dst_slots,
    Tensor copy_counts
) -> ()
```

方法无返回 tensor；`hbm_k_rope` 和 `hbm_kv_cache` 保持原地写入语义。

### 5.4 A2/A3 实现

A2/A3 构建清单增加以下自定义算子：

- `lightning_indexer_decode_update`
- `kvcache_scatter_copy`
- `sparse_flash_attention_for_offload`
- `kv_cache_full_block_dump`

算子 kernel、tiling、op host 和 torch adapter 来自原 DSA 分支；只改造 v0.23.0 的 CMake、ACLNN 打包和 torch binding 接入。

### 5.5 A5 实现

A5 不编译仅支持 A2/A3 指令架构的四个 DSA 自定义 kernel。运行时处理如下：

1. 通过 `torch.library.Library("_C_ascend", "FRAGMENT")` 注册与 A2/A3 完全相同的 LIDU/KSC schema。
2. LIDU 调用 A5 原生 `torch_npu.npu_lightning_indexer`，再用 device tensor 更新 resident map 和输出 buffer。
3. KSC 使用 device-side gather 和 `index_copy_` 实现 miss-only 物化。
4. Full-block dump 使用 block 级 `index_select`/`index_copy_`。
5. SFA-Offload 使用 resident gather、FP32 score/softmax 和 tensor einsum。

A5 路径保持功能接口，但不是 A2/A3 融合 kernel 的性能等价实现。上线前必须验证峰值显存、TPOT 和长上下文稳定性；如性能不达标，后续应在不改变公开 ABI 的前提下替换成 A5 融合 kernel。

## 6. v0.23.0 框架适配点

| 模块 | 改动 | v0.23.0 适配目的 |
|---|---|---|
| `vllm_ascend/core/kv_cache_interface.py` | 新增 `IndexerKVSpec` | 让 Indexer cache 成为独立 KV group |
| `patch/dsa_sparse/patch_kv_cache_utils.py` | split group 容量规划 | 绕开原生统一 page-size/common-block 假设 |
| `patch/dsa_sparse/patch_kv_cache_decoupling.py` | `MultiBlockPool` 和独立 coordinator | Indexer/MLA block id 不混用；保持 v0.23 `allocate_slots` 签名 |
| `patch/dsa_sparse/patch_scheduler.py` | barrier、阶段推进、MTP trim、dump metadata | 适配 v0.23 scheduler/preempt/recompute 生命周期 |
| `patch/dsa_sparse/patch_request.py` | request sidecar | 保存 stage、resident budget、block hash delta |
| `patch/dsa_sparse/patch_scheduler_output.py` | 扩展 v0.23 输出数据 | 跨 EngineCore/worker 传递 DSA 元数据 |
| `dsa_sparse/dsa_model_runner_adapter.py` | 包装原生 `_update_states` | 继续使用 v0.23 add/remove/condense/reorder |
| `worker/npu_input_batch.py` | 增加 `dsa_state` | 在最终行顺序上保存 tensorized sidecar |
| `worker/model_runner_v1.py` | split cache 分配、slot mapping、metadata | Indexer 用 dense table；MLA 用 resident table |
| `attention/sfa_v1.py` | sparse decode 接入 | 在 v0.23 SFA decode 分支插入 LIDU → KSC → SFA-Offload |
| `ops/mla.py` | 保存 Indexer cache layer name | split cache 后仍能定位 dense Indexer tensor |
| `worker/worker.py` | worker-local manager/DRAM 初始化 | NPU 设备选定后创建每 rank 独立资源 |
| `platform.py` | 配置物化和启动校验 | 在 cache planning 前启用补丁并 fail-fast |
| `vllm_ascend/__init__.py` | EngineCore child bootstrap | 多进程子进程仅在 DSA 启用时安装运行时补丁 |
| `CMakeLists.txt`、`csrc/build_aclnn.sh`、`csrc/torch_binding.cpp` | 编译与注册 | A2/A3 融合算子；A5 fallback 宏和同名 schema |

补丁安装是幂等的。未启用 `dsa_sparse_config.enabled` 时，不安装 DSA scheduler/KV planning 数据面补丁，普通 v0.23.0 请求继续使用原生路径。

## 7. 推测解码边界

迁移来源分支明确未集成 speculative/MTP decode。该模式会让一个 scheduler
step 写入多个候选 token，而 DSA 的 split Indexer/MLA cache、resident window 和
DRAM full-block dump 尚未完成 accepted/rejected token 的设备侧精度验收。它可能
不触发设备异常，却返回错误 token。

因此本分支在 `validate_dsa_sparse_runtime_config()` 中拒绝所有非空
`speculative_config`，并提示删除 `--speculative-config`。只有在 A2/A3/A5 上完成
dense 与 sparse 的逐 token 对比、draft 接受/拒绝和 block 边界回滚验收后，才应
重新开放该组合。

## 8. DP+TP 资源与一致性

| 资源 | 隔离范围 |
|---|---|
| Scheduler request stage/预算 | 每 DP EngineCore 独立 |
| Indexer/MLA `BlockPool` | 每 EngineCore/worker cache manager 独立 |
| Resident row pool | 每 worker 进程、每 NPU 独立 |
| Hot DRAM arena | 每 TP worker/NPU 独立 |
| LIDU/KSC output buffer | 每 worker 进程独立 |
| Stream 顺序 | 当前 worker 的当前 NPU stream |

TP group 中每个 rank处理相同请求顺序，并从 scheduler output 获得相同 logical block/hash delta；物理 HBM/DRAM tensor 仍是 rank-local。DP rank之间不共享请求或 DRAM block table。

容量按每个 DP rank 的 `max_num_seqs` 计算，不额外乘以 `data_parallel_size`。每台机器必须为其本地 DP×TP worker 分别预留 HBM 和 DRAM。

## 9. 配置

入口为：

```json
{
  "dsa_sparse_config": {
    "enabled": true,
    "split_indexer_cache": true,
    "indexer_mla_block_ratio": 3
  },
  "ascend_compilation_config": {
    "enable_npugraph_ex": false
  }
}
```

### 9.1 公开参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `enabled` | `false` | 是否启用 DSA sparse offload |
| `split_indexer_cache` | 启用 DSA 时强制 `true` | Indexer/MLA 是否拆分 |
| `indexer_mla_block_ratio` | `3` | KV planning 的 Indexer:MLA block 容量权重 |
| `sparse_activation_tokens` | `6144` | 进入 sparse decode 的最小 context |
| `prompt_budget_thresholds` | `[32768, 65536]` | 按 prompt/context 选择 resident budget 的阈值 |
| `resident_budget_tokens` | A2/A3: `[6144,10240,12288]`; A5: `[6144,8192,8192]` | 各长度档位的 resident token 数 |
| `max_active_reqs` | `256` | resident/DRAM request row 预分配上限，必须不小于 `max_num_seqs` |
| `hot_cpu_block_multiple` | `3` | DRAM block 数相对 Indexer HBM block 数的倍数 |
| `enable_row_mode_decode_graph` | `false` | 本次 eager 范围必须为 `false` |
| `trace_points` | 未启用 | 可选调试 trace 配置 |

### 9.2 固定要求

- `--block-size 128`
- 模型和 KV cache 使用 BF16/FP16
- `--enforce-eager`
- `--no-enable-chunked-prefill`
- `--no-enable-prefix-caching`
- `ascend_compilation_config.enable_npugraph_ex=false`
- 不配置 KV connector/offloading connector
- `pipeline_parallel_size=1`
- `decode_context_parallel_size=1`
- `prefill_context_parallel_size=1`

## 10. 构建与部署

v0.23.0 依赖基线为 `torch==2.10.0`、`torch-npu==2.10.0.post2`、`transformers==5.5.4` 和 `triton-ascend==3.2.1`。CANN、驱动和固件版本应与该依赖组合及目标服务器发布矩阵一致。

### 10.1 源码安装

```bash
git submodule update --init --recursive

# A2 示例；A3 使用 ascend910_9391；A5 使用服务器实际 ascend950* 型号。
export SOC_VERSION=ascend910b1

pip install --no-build-isolation -e .
```

`COMPILE_CUSTOM_KERNELS` 默认值为 `1`。A2/A3 安装会通过 `csrc/build_aclnn.sh` 打包新增的四个 DSA ACLNN 算子；A5 不编译这四个 A2/A3 kernel，运行时使用同名 eager 适配。

### 10.2 启动

仓库提供 `examples/glm51_dsa_sparse_mtp.sh`：

```bash
MODEL_PATH=/models/GLM-5.1 \
DP_SIZE=2 \
TP_SIZE=8 \
MAX_NUM_SEQS=8 \
MAX_MODEL_LEN=65536 \
bash examples/glm51_dsa_sparse_mtp.sh
```

该文件保留旧文件名以兼容现有部署，但不再传入 `--speculative-config`。量化模型
默认使用 `QUANTIZATION=ascend`；BF16 权重应显式传入 `QUANTIZATION=`。

如果完整 prompt 可能达到 `MAX_MODEL_LEN`，`MAX_NUM_BATCHED_TOKENS` 必须至少覆盖该 prompt；否则因关闭 chunked prefill，请求会被调度配置拒绝。

## 11. 已执行的离线检查

```bash
python -m unittest tests.ut.dsa_sparse.test_v023_migration_contract -v
python -m compileall -q vllm_ascend tests/ut/dsa_sparse
git diff --check
bash -n csrc/build_aclnn.sh examples/glm51_dsa_sparse_mtp.sh
```

迁移契约测试覆盖：

- ACL 算子名不变。
- LIDU/KSC torch schema、方法名和参数顺序不变。
- A2/A3 编译清单与 A5 fallback 分流。
- A5 数据面无 Host `.item()` 和空槽负索引污染。
- v0.23.0 `KVCacheManager.allocate_slots` 签名。
- Indexer logical BF16/FP16 cache spec。
- 最后 prefill 的完整 block 全量 dump。
- speculative/MTP 启动期 fail-closed 和示例禁用约束。
- 多进程 Worker 在模型初始化前安装 Indexer spec patch。
- scheduler decode barrier 不丢失临时 preempt 请求。
- 非 DSA EngineCore child 不安装 DSA 数据面补丁。
- DRAM/resident ownership 只接受 MLA spec。
- worker 初始化缺少 Indexer/MLA tensor 时 fail-fast。

## 12. 真机验收矩阵

当前环境不能执行本节。发布前至少完成以下验收：

| 维度 | 用例 | 通过标准 |
|---|---|---|
| 构建 | A2、A3 分别从干净环境 `pip install -e .` | 四个 DSA ACLNN 算子编译、安装、加载成功 |
| 启动 | A2/A3/A5，DP1×TP、DP2×TP | 无 cache group、schema、设备映射或 DRAM arena 错误 |
| ABI | 直接调用 LIDU/KSC schema | 参数顺序、dtype、shape、原地输出与来源分支一致 |
| Dense 基线 | 关闭 DSA，固定 prompts/seeds | 与未改 v0.23.0 token 输出一致 |
| Sparse 精度 | 开启 DSA，6K/10K/12K 或 A5 6K/8K budgets | 逐 token 对比 dense 基线，误差符合 BF16/FP16门限 |
| 长上下文 | prompt 覆盖 6K、32K、64K 及服务上限 | 无错误 block、越界、提前释放或 DRAM OOM |
| Speculative guard | 启用 DSA 并传入任意 `--speculative-config` | 启动期明确拒绝，不进入推理数据面 |
| Continuous batching | 请求加入、完成、preempt、recompute、condense | 无串请求、resident row 泄漏或 block refcount 泄漏 |
| DP+TP | 多 DP rank并发不同长度请求 | rank资源隔离；TP rank logical mapping 一致 |
| A5 稳定性 | 长时间压测和高 miss-rate | 无 Host 同步、OOM、NaN 或地址错误 |
| 性能 | TTFT、TPOT、吞吐、HBM/DRAM 带宽 | 达到项目验收门限；A5 特别记录组合实现开销 |

建议在 debug 验收中记录每个 DP/TP rank 的设备号、resident rows、HBM/DRAM block 使用率、LIDU miss rate 和 full-block dump 数量。

## 13. 已知限制与风险

1. A5 当前是接口兼容的 eager 组合实现，不是融合 kernel；功能和性能必须真机确认。
2. DRAM arena 按 layer、cache type、worker 固定预分配。`hot_cpu_block_multiple`、`max_active_reqs`、模型层数和上下文上限会显著影响内存。
3. `indexer_mla_block_ratio=3` 是容量规划权重，不代表所有部署都应使用同一值；修改后需要重新测量 HBM/DRAM 和并发上限。
4. Speculative/MTP、prefix cache、chunked prefill、graph、DCP、PCP、PP 和通用 KV offload 不在本次正确性闭包中。
5. 当前契约测试是源码级和轻量逻辑测试，不能替代自定义 kernel 编译、设备数值和多机通信验收。

## 14. 回滚方式

最小回滚方式是去掉 `additional_config.dsa_sparse_config`，或设置：

```json
{
  "dsa_sparse_config": {
    "enabled": false
  }
}
```

此时不启用 split cache、scheduler 状态机、DRAM hot store 和 sparse attention 数据面。编译产物中可以保留新增算子注册，不影响普通 v0.23.0 模型路径。

若需要代码级完全回滚，应同时删除：

- `vllm_ascend/dsa_sparse/`
- `vllm_ascend/patch/dsa_sparse/`
- 四个新增 `csrc/attention/` 算子目录
- `examples/glm51_dsa_sparse_mtp.sh`

并还原本说明第 6 节列出的 v0.23.0 接入文件。

## 15. 升级维护规范

后续 rebase 到新框架版本时，优先检查以下稳定边界：

1. `KVCacheManager.allocate_slots` 的完整签名和返回语义。
2. `SchedulerOutput`、`NewRequestData`、`CachedRequestData` 字段。
3. scheduler 的 preempt、free 和 recompute 清理顺序。
4. `NPUInputBatch` add/remove/condense/reorder 后的最终行顺序。
5. KV group planning 是否仍允许 Indexer/MLA 独立 page size、block count 和 pool。
6. `AscendSFAImpl` decode metadata、Indexer cache layout 和 A5原生 lightning indexer schema。
7. EngineCore 多进程 bootstrap 是否仍能在 child 初始化前获得完整 `VllmConfig`。
8. A2/A3/A5 的 CMake 架构宏、ACLNN schema 和 torch PrivateUse1 注册。

任何一项变化都应先更新 `tests/ut/dsa_sparse/test_v023_migration_contract.py`，再调整实现，避免通过整文件覆盖破坏 v0.23.0 新增能力。
