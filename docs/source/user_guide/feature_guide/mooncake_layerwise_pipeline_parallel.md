# Mooncake layerwise + pipeline parallel 适配与验收

实现基于 `Eric-dot/vllm-ascend:mooncake-layerwise-pp-dcp`，基线提交为
`ba7bc4d4587ad8cbbadb644094ea5c0bcddb8a7e`，本地分支为 `codex/mooncake-layerwise-pp`。
这是可以上机验证的代码适配；本地没有 Ascend NPU、模型权重和 Mooncake 服务，尚不能宣称模型精度及实际性能通过验收。

## 适配范围

| 配置 | 行为 |
| --- | --- |
| 单 KV group，TP + PP | 支持 stage 独立对象；PP 可以非均匀划分 |
| 多 KV group / hybrid attention，TP + PP | 支持局部层映射、多 cache entry、每组独立 commit 和可达性 mask |
| MLA / 复制的 KV head | 保留每个 stage 内 `tp_rank % put_step == 0` 单写者，其他 TP rank 读同一对象 |
| 额外 draft/MTP cache layer | 生命周期计数覆盖本 stage 已注册的额外物理层；实际模型 callback 次数需要上机确认 |
| producer/consumer 同 TP、同 PP partition、同模型与 cache 配置 | 可复用 |
| producer PP=2、consumer PP=1，或 PP partition 不同 | 命名空间不同，产生 miss；不做跨拓扑重排 |
| 某 stage 的投影 group 为空，包括 draft-only group | 启动失败；需要后续设计 group-owner manifest 后再支持 |
| DCP、PCP、P/D TP mismatch、hybrid Mamba state | 维持明确拒绝 |
| PP=1 | 保留原有 single-group 和 hybrid key 格式 |

分支名包含 DCP，不代表本次修改实现了 DCP。DCP 涉及 token shard 与物理 token 排列，不能仅给 key 加 rank。
同样，draft layer 的 CPU 字节回环不等于所有 MTP 模型都已通过精度验证。

## 三个 PR 的借鉴方式

- [#16747](https://github.com/vllm-project/vllm-ascend/pull/16747)：保留 hybrid 分组、reachability mask、真实 block stride、range/session 数据面和 attention window。
- [#16208](https://github.com/vllm-project/vllm-ascend/pull/16208)：借鉴 PP 下区分全局层、局部层和完成生命周期的原则。Memcache GVA 的全局地址空间不能直接代入 Mooncake stage 独立对象。
- [#17037](https://github.com/vllm-project/vllm-ascend/pull/17037)：借鉴 stage 隔离与 all-stage hit check。其公开修改中已经包含 global/local 映射的修补；用户报告的硬件精度问题还需要日志与运行配置才能归因。本实现不是对该 PR 全部改动的 cherry-pick。

## 为什么仅放开 PP 限制不够

### 1. 三种层编号必须分开

例如 7 层模型按 `3,4` 划分，stage 1 的层为：

| 模型全局层 | stage 局部层 | group 0 的局部索引 | group 1 的局部索引 |
| --- | --- | --- | --- |
| 3 | 0 | 0 | — |
| 4 | 1 | — | 0 |
| 5 | 2 | — | 1 |
| 6 | 3 | 1 | 2 |

基线 `_init_layerwise_config` 将 group 映射建立在 **stage 局部层** 上，
但 `process_layer_data` 使用 `local_layer + pp_layer_offset` 查询。
以上布局中，局部层 0 误查键 3，会取到全局层 6 的 group 和 group 内索引；
这不一定越界，也可能把其他层的合法字节写到错误位置。

现在 Mooncake 的任务与 event 使用 stage 局部层；查询得到的 `layer_idx_in_group`
才传给 range builder。多组映射缺失直接失败，禁止回退为 `(group 0, local_layer)`。
`LayerBatchBuilder.build()` 的直接调用路径也统一使用 group 内索引。
其他后端的 group 查询保留原行为。

### 2. 每个 stage 的对象从偏移 0 开始

每个 `(block, group, stage, saving head)` 是独立对象，大小为该 stage/group 实际 cache entry 的
per-block byte length 之和。对象内偏移按这些 entry 的前缀和计算，源/目的地址按真实 block stride 计算。
不使用 `全模型层数 × 平均每层大小`，也不叠加全局 PP layer offset。

single-group PP key：

```text
model@mooncake_pp_v3:<digest>@pp_rank:<stage>@<block-hash-or-tail>@<head>
```

hybrid PP key：

```text
model@mooncake_hybrid_v1:<layout-digest>@pp_rank:<stage>@group:<id>@block:<tokens>@<hash>@<head>
```

公共 PP namespace 包含真实 partition、TP size、vLLM ModelConfig/CacheConfig hash、cache dtype、
归一化后的 block size 和 speculative config hash。CacheConfig hash 也在归一化后的配置副本上计算。
partition 使用 vLLM 的层划分 API，包含自定义非均匀划分。
hybrid layout 再加入按全局 group 顺序排列的 block-size signature。
PP 下不散列各 worker 不同的局部 `layer_names`；PP=1 继续使用原来的 membership digest。

这不是权重内容校验。同一路径被覆盖为不同 checkpoint 时，仍需要隔离/清理旧 pool；各实例必须使用一致代码与权重。

### PP layerwise 零命中修复（2026-09-22）

`dc3c026` 的 PP namespace 存在初始化阶段配置不一致的问题：

1. worker 保留启动时的 `CacheConfig.block_size`。
2. vLLM EngineCore 创建 KV groups 后，将 scheduler 的 `block_size` 改成各组的最小值。
3. DeepSeek-V4 的 SWA、compressed KV 和 compressor state 使用不同的 block size。
   例如 CLI 为 32，而 state group 可以为 2，因此 scheduler 与 worker 的配置值不同。
4. 旧 namespace 同时散列原始 `block_size` 和包含它的 `CacheConfig.compute_hash()`，
   导致 worker PUT key 和 scheduler lookup key 前缀不同。即使所有 stage 都成功提交，对象仍查不到。

普通池化不使用这套新增的 PP namespace，因此 `use_layerwise=false` 可以正常命中。
这也解释了为什么问题不能通过增加预取层数或修改 attention window 解决。

修复时从实际 `KVCacheConfig.kv_cache_groups` 提取共同的最小 block size，
在配置副本上同时归一化直接散列值和 CacheConfig hash；不修改模型运行配置。
namespace 升级到 `mooncake_pp_v3`，PP=1 key 不变。
新增回归通过生产 worker 的 range/session 路径写入两个 stage，再调用生产 scheduler 查询：
旧实现返回 miss，修复后返回 hit。另一测试覆盖 CacheConfig hash 中的 block size、dtype 隔离及配置不被修改。

本次 CPU/mock 验证：针对性测试 `41 passed, 4 subtests passed`；扩大回归为
`419 passed, 4 failed, 240 subtests passed`，失败仍为下文列出的四项既有 mock/依赖问题。
这四项已在独立导出的未修改 `dc3c026` 上再次复现。
`test_layerwise_cache_layout.py` 因缺少真实 vLLM 依赖未纳入该运行。
Ruff 和 Python 编译检查通过；执行 `bash format.sh ci` 时因环境缺少 `pre-commit` 提前退出，完整格式检查未通过验收。
尚未在真实 A5、DeepSeek-V4 权重和 Mooncake 服务上验证修复后的命中率、精度及性能。

服务器复测步骤：

1. 更新并重启所有相关 scheduler/worker，保持原来的 PP、DP、MTP 和 block-size 参数。
   不要把启动参数 `--block-size 32` 改成 state group 的大小。
2. 从启动日志提取 `Mooncake PP`：scheduler 和各 PP worker 的 `namespace` 必须相同；
   `config_block_size` 可以不同，`group_block_sizes` 应对应相同分组。
3. 新 namespace 不读取旧版对象，第一轮需要重新写入；等预热请求完成后再发送相同前缀，检查 external hit。
4. 命中恢复后，继续用下文精度脚本对比完整生成 token IDs；仅输出 1 token 的性能测试不能代替精度验收。

### 3. hit 必须覆盖全部 stage 和 saving head

scheduler 从 stage 0 的投影配置出发，为每个需要的 group/block 枚举所有 PP stage 和 saving head。
只有 `batch_is_readable` 全部返回 True，才向 reachability coordinator 报告该 block 命中。
不把已分配、仍在写入的对象视为可读。任意 stage/group PUT 失败，都不能使不完整的逻辑状态成为 hit。
MLA 不要求所有 TP replica 各写一份，只要求所有独立 saving head。

当 group 在某个 stage 为空时，scheduler 不能从自身局部配置推断远端 group 的 owner 集合。
当前实现对此 fail-fast，包括 speculative decode 场景，避免用 warning 掩盖永久 miss。

### 4. 完成计数覆盖本地 draft layer，并等待空末层之前的传输

PP 的基础模型层数不包含后注册的 draft cache。Mooncake 按本地实际物理层数建立任务及完成计数，
避免最后一个 group 在 MTP/draft layer 到来前就被视为完成。

还有一个可独立复现的竞态：最后一层没有可保存的 range 时，计算线程会直接设置它的完成 event，
但此前 PUT 可能还在后台队列。如果只等末层 event，源 block 的生命周期可能提前结束。
现在在 step 末尾同时确认此前本地层的完成 event，然后才允许结束保存。

没有增加逐层全局 barrier 或逐层全队列 drain。原有预取窗口、最多 8 个未完成 send task 的策略、
逐组 commit 及 range batching 保留。CPU 延迟 PUT 测试同时要求计算能推进到末层、step 不能提前返回。
实际 HCCL 与 Mooncake 网络争用仍需 trace 验证。

## 本地验证

新增 `tests/ut/distributed/ascend_store/test_mooncake_pipeline.py`，通过真实 worker 初始化、
session tracker、range builder 和发送/接收线程 handler，在 CPU buffer 上验证：

- `PP=2, TP=2`，`3/4` 非均匀划分，single-group 与 hybrid，以及末 stage 额外 draft layer。
- 每个 stage/head/layer/block 使用不同数据；销毁原数据后恢复到不同 block ID，逐字节比较。
- 非连续 block stride、同一层多个 cache entry、不同 group page size、SWA sparse mask、null block 不被写入。
- 真实 scheduler 与各 stage 的 namespace/layout 一致；缺少任何 stage/head 时为 miss。
- MLA replica 单写者；某 stage 某 group PUT 失败时撤销该组对象，禁止完整命中。
- layout 隔离、空 group/missing mapping 拒绝、PP=1 key 兼容。
- 延迟前层 PUT + 空末层，证明既保留层间重叠又不提前结束源 buffer 生命周期。

将层查询临时恢复为原先的“局部层 + PP offset”，新增映射测试失败；
将最终等待临时恢复为“只等最后一层”，延迟 PUT 测试失败。两项反向验证均已执行。

本地运行使用 Windows Python 3.11、CPU buffer 与仓库 `_mock_deps`。
测试进程补充了 A3 build-info stub 和未参与本次测试的 KVPP placement mock；没有真实 NPU/vLLM runtime。
最终结果：新增 9 项测试及 4 个参数子用例通过；扩大回归在排除下述 4 项已复现的环境失败后，
`417 passed, 4 deselected, 240 subtests passed`。Ruff、Python 编译和脚本 `--help` 检查通过。
标准 Linux 开发环境可以运行：

```bash
pytest -q tests/ut/distributed/ascend_store/test_mooncake_pipeline.py \
  tests/ut/distributed/ascend_store/test_mooncake_hybrid.py \
  tests/ut/distributed/ascend_store/test_mooncake_layerwise.py
```

扩大 AscendStore CPU/mock 测试时，另有 4 个环境相关失败，已在**未修改的同一基线**复现：
MambaCopyBuffers mock spec、两项 coordinator mock 返回值、KVPP 缺少真实 torch/vLLM 工具模块。
`test_layerwise_cache_layout.py` 需要当前环境没有的真实依赖，未纳入该本地运行。
这些结果不能替代 Linux 原生依赖下的完整 CI。

## NPU 精度验收：三组独立对照

新增脚本 `tests/e2e/nightly/single_node/models/scripts/mooncake_pipeline_correctness.py`。
使用三个独立进程，配置完全相同，并在每轮清空本地 prefix cache：

1. `baseline`：不加载 connector，重复重算，先确认模型自身结果稳定。
2. `save`：仅写 Mooncake，每一轮都与独立 baseline 的完整生成 token IDs 比较。
3. `load`：仅读 Mooncake，要求每个请求都有 cache hit，且完整 token IDs 与 baseline 一致。

这样不会因为“冷、热两次走了相同错误路径”而误判通过。JSON 报告保存 token IDs、命中 token 数、
每轮耗时、可用时的 TTFT、输出 tokens/s 和失败原因。它验证生成一致性，不声称验证了完整 logits。

先准备支持 range/session API 的 Mooncake client、常驻 Master 和存储 segment。
producer 进程退出后若其 segment 随之消失，load 测试会 miss；需保证独立可存活的存储容量。

```bash
export MOONCAKE_CONFIG_PATH=/path/to/mooncake.json
SCRIPT=tests/e2e/nightly/single_node/models/scripts/mooncake_pipeline_correctness.py
MODEL=/path/to/model
SALT=pp-acceptance-20260921-001
EXTRA='{"dtype":"bfloat16"}'  # 三次必须一致；按实际模型加入 quantization / speculative_config

python "$SCRIPT" --model "$MODEL" --tp 2 --pp 2 --enforce-eager \
  --prompt-salt "$SALT" --engine-args "$EXTRA" --mode baseline --output /tmp/pp-baseline.json
python "$SCRIPT" --model "$MODEL" --tp 2 --pp 2 --enforce-eager \
  --prompt-salt "$SALT" --engine-args "$EXTRA" --mode save \
  --reference /tmp/pp-baseline.json --output /tmp/pp-save.json
python "$SCRIPT" --model "$MODEL" --tp 2 --pp 2 --enforce-eager \
  --prompt-salt "$SALT" --engine-args "$EXTRA" --mode load \
  --reference /tmp/pp-baseline.json --output /tmp/pp-load.json
```

prompt 长度必须超过各 group transfer granularity 的 LCM；默认 32769 token 不是适用所有模型的固定要求。
若目标模型不支持 4K chunked prefill 或默认 block size，使用 `--engine-args` 设置模型要求的值。
先跑 eager；图模式、MTP、DP 和多机场景需要各自重复验收，不能沿用 eager 的结论。

精度验收顺序建议：TP=1/PP=1 → TP=2/PP=2 → 实际部署拓扑；
覆盖完整块和边界长度、部分命中、chunked prefill、重复 prefix、并发请求、preemption/recompute、
group 传输失败及 Master 重启。模型任务集上再跑无 pool / pooled 的同样精度评测。
若 baseline 自身不稳定，应先测量重算路径差异，不能直接把 token 不同归因于 KV 传输。

## 性能验收

实现从结构上保留 overlap；本地没有测得任何 NPU 加速数字。
上述离线脚本的耗时只作为 smoke 指标，不作为生产吞吐结论。

在同一 TP/PP、模型、负载和 pool 容量下，比较无 connector、现有非 layerwise Mooncake、本实现：

| 维度 | 必须记录 |
| --- | --- |
| 请求 | 输入/输出长度、并发、请求率、命中 token 比例、chunk 大小 |
| 延迟 | TTFT P50/P90/P99、TPOT/ITL、端到端延迟 |
| 容量 | requests/s、output tokens/s、NPU/HBM、网络吞吐、Master CPU/RPC rate |
| trace | PUT/GET 与 attention 的重叠、HCCL/PP send-recv 争用、每组 commit 时间、stage 尾部等待 |
| 压力 | 冷 cache、热 cache、混合 hit/miss、并发重复 prefix、长时间运行 |

用相同随机种子和请求集，剔除预热，重复运行；精度通过后再调整
`layerwise_prefetch_layers`、`layerwise_max_transfer_blocks`、`layerwise_max_transfer_bytes`。
正常性能运行关闭 `VLLM_ASCEND_KVPOOL_RANGE_DEBUG`。发现 step 尾部等待时先检查真实网络瓶颈，
不能通过取消等待来换取表面吞吐，否则源 block 提前复用会重新引入竞态。

当前结论：地址映射、PP key 完整性和完成时序已有本地可复现验证；
真实模型精度、图模式、MTP callback、HCCL/Mooncake 传输与性能仍待上述硬件验收。
