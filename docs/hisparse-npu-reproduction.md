# HiSparse 四算子 A3 UT：执行与复现记录（2026-11-02）
>
> 提交前复验（2026-11-02）：统一导入入口及 Ruff 整理后，A3 145 passed（164.16s），
> CPU 140 passed、8 个 NPU 用例 deselected（23.03s）。最终日志、XML、源码哈希及
> 检查边界见 [validation evidence](hisparse-validation/2026-11-02/README.md)。下文保留各阶段执行记录。

## 最终结果与边界

在 **130、板卡4/芯片0/物理设备8** 上完成真实 NPU 执行：

- **145 passed，0 failed，0 skipped，0 xfailed，0 deselected**，pytest 用时 **21.58秒**。
  其中新四算子 NPU 套件119项，原 invalidation 套件26项。
- 修复设备编译问题后，CPU Triton interpreter 回归为 **140 passed，8 deselected**，
  用时20.08秒。8项 NPU 专用用例已在上述 NPU 套件实际执行通过。
- `test_hisparse_invalidation_covers_every_index_group` 在真实 NPU 上通过。
- vLLM 参考工作树未改动；vLLM-Ascend 的实现、测试、脚本和文档均保留未提交状态。

这里的“上游用例通过”指**移植到 Ascend 的四算子层场景通过**，不是把 CUDA/ROCm
门控的整个 `test_sparse_mla_backends.py` 原样运行成功。没有运行模型、FlashMLA/
FlashInfer attention、backend runtime 集成、KV writeback/row DMA 集成或性能基准。
原始文件的 CUDA shared-memory 容量拒绝条件不适用于本实现的全局 workspace。
不能将这一 CUDA 专用拒绝断言解释成 Ascend 应有的容量限制。

## 锁定环境

工作区：`/mnt/share/h00954440/HiSparse-Ascend-opsWorkWithCodex`。
两个仓库分支均为 `HiSparse-Ascend-opsWorkWithCodex`。

| 项目 | 实际记录 |
| --- | --- |
| vLLM | `d5301a4f5487db1d7e8d87daf7828225b95b1774` |
| vLLM-Ascend | `cae7ff0290dd6c6155f71d7223e0fd4aff95c9e9` + 未提交改动 |
| Host | 130 / `node-97-31`；129 的16个设备当时均有 vLLM 计算进程 |
| 可见设备 | `ASCEND_RT_VISIBLE_DEVICES=8`；PyTorch device_count=1，型号 `Ascend910_9382` |
| 已有运行容器 | `vaws-weiguihua-20261030`，只启动本任务限时子进程，保留其运行状态 |
| 镜像 ID | `sha256:8725ad72b9db273848f13894bad0e547a67567cccb2f8692b7e3588b2e959982` |
| Python | `/usr/local/python3.12.13/bin/python3`，3.12.13 |
| 依赖 | torch `2.10.0+cpu`，torch-npu `2.10.0.post7`，triton-ascend `3.6.0`，pytest `8.4.2` |
| SDK / 驱动 | 容器 CANN_VERSION=`9.2.0`；Host npu-smi/driver=`26.1.0` |

torch 的版本后缀 `+cpu` 不代表本轮执行在 CPU：NPU 由 torch-npu 提供；测试输入位于
`npu:0`，公开 NPU wrapper 启动真实编译 kernel。CPU interpreter 是另一次独立运行。
没有导入或替换整个 vLLM 平台为 CUDA，也没有引用 Ascend PR #19109。

## 设备和运行环境选择过程

1. 129 的全部设备都有运行中的 `VLLMWorker_DP`，予以保留。
2. 130 当时没有 NPU 计算进程，但新建单卡容器无法初始化。逐卡检查全部16个设备，
   驱动均报 `Resource_Busy(EL0020): Device ... is occupied by multiple containers`。
   “无计算进程”与“新容器可访问”是两个不同检查。
3. 尝试对板卡4开启临时 device-share，驱动说明 A3 按整卡设置，并拒绝开启。
   恢复检查确认两个芯片均为 `False`。没有重置卡，没有停止或删除其他用户容器。
4. 自有 `h00954440-workspace` 的设备9同样初始化失败；该容器恢复为停止状态且保留。
5. 已有运行容器中，限定物理设备8后初始化成功。后续只操作本任务进程和
   `/mnt/share/h00954440` 文件，不安装依赖、不改容器环境或既有服务。
6. 用户后续明确要求本次不向所有人 `wall`，之后批次使用聊天提示和本地日志，
   不再广播。`run_hisparse_npu_tests.sh` 不调用 `wall`。

本轮新建的诊断/CPU 容器与 timer 已清理；使用已有容器时不停止或删除该容器。
最终检查没有本任务 NPU 进程，板卡4 device-share 两芯片保持 `False`。

## Host 映射与真实搬运

`tests/e2e/pull_request/one_card/test_hisparse_ops_npu.py` 中的 `RegisteredHost` 是
**测试专用**映射：先选择当前 NPU context，分配页对齐的 mmap Host 存储，调用
`aclrtHostRegister(..., ACL_HOST_REGISTER_MAPPED, &device_ptr)`，再把设备可见地址
通过 `HiSparseHostMapping` 协议注入生产 `HiSparseTritonCacheOps`。

`address_of` 校验注册存活、CPU 连续视图和范围，并加上子视图偏移。测试完成后先同步，
再 `aclrtHostUnregister`。注册/注销均返回0。生产 gather kernel 直接读这一映射地址；
没有把 Host payload 预搬到 NPU 来伪装 Host gather，也没有普通 CPU 指针回退。
初始化预期值、构造输入和读取断言的 CPU/NPU copy 只存在于测试层。

测试注册成功与真正的 gather payload 比较均在这张 A3 上完成，但当前生产 backend
仍需实现映射分配/注册/生命周期和 workspace 接入；本次没有把测试映射注册成 backend。

## 上游四算子场景覆盖清单

来源：锁定 vLLM 的 `tests/v1/attention/test_sparse_mla_backends.py`。
命名映射如下；除明确列出的算子提取边界外，保留其关键参数和断言。

| 上游名称（省略 `test_`） | 实际 NPU 场景 |
| --- | --- |
| `hisparse_uses_graph_stable_request_state_mapping` | K=1、H=2、block64、state=[1]；state0不变，state1包含Host行64 |
| `hisparse_maps_speculative_rows_through_request_state` | 4行同 request，K=1、H=2、resident block64；全部输出64 |
| `hisparse_speculative_rows_resolve_host_misses_consistently` | 原三组 `(rows,stride,K)=(8,128,128),(4,32,128),(8,2048,2048)`；4步，resolver→compact gather；逐项 metadata 和 payload 比较 |
| `hisparse_speculative_rows_count_a_shared_load_once` | K=128、block64、H=256、两行相同 top-k；stats=[128,128]、swap_counts=[128,0] |
| `hisparse_resolver_reads_current_request_ids` | 在同一组 NPU 输入地址上更改 request IDs 和 valid-row 标量；验证 padding 与当前输入 |
| `hisparse_union_table_scales_with_union_bound` | K=128/8192，H=32768，真实 resolve+gather；不采用上游 CUDA shared-memory skip |
| `hisparse_expanded_hash_keeps_union_bound` | K=4、H=4、union=3；子进程先执行合法输入，再只把第4项改成有效，验证实际设备 trap |
| `hisparse_resident_rows_bypass_hot_lru` | 原 K=4、H=5、block64、top-k=[1,2,3,4]；输出[65,66,67,68]、valid_count=4、LRU不变 |
| `hisparse_swap_in_preserves_rows_across_eviction` | 原4 request、K=128、H=256、block64，5步(0,0,1,2,0)，末列-1；256/528/656字节，累计统计与 payload 精确比较 |
| `hisparse_multi_step_swaps_match_independent` | 4 request、K=128、H=384、两轮各两步；真实 producer/independent resolver、shared-layer gather replay；比较索引和整个 cache |
| `hisparse_multi_step_writes_request_major_output` | 2 request、query_len=3、K=4、H=12、block64；strided attention/count 输出及请求主序 payload |
| `hisparse_remaps_strided_hma_rows_for_attention` | block4、3倍page stride、非零offset；访问第二page；比较 attention 编码和整个带保护字节的 storage |
| `hisparse_gather_zeroes_unaligned_destination` | 原16-byte rows、33-byte storage、stride17、源1/目标1；清零并保留guard=7 |
| `hisparse_gather_rejects_short_attention_block_stride` | 原block2、stride1，在公开 wrapper 拒绝；异常类型为适配层 ValueError |
| `hisparse_newest_write_and_recycled_slot_invalidation` | 提取四算子链：resolve→compact→shared plan→重写Host行→两组invalidation→重新resolve/gather；确认新payload |
| `hisparse_invalidation_covers_every_index_group` | 原始表、request/state映射、padding、未列出表保护；通过原移植文件实际执行 |
| `hisparse_gather_prefill_cache_prefers_resident_rows` | 原block table [[5,2,0],[9,3,0]]、lengths=[5,8]、source block4/resident block2；16 float32、528/656 uint8；独立构造算子plan，验证预填resident行不被Host覆盖 |

以上17个函数对应**24组原参数场景**。此外增加可选输出64组合、非法源/目标、count越界、
dtype位模式、page gap/子视图、uint64计数、padding、图重放和stream/event等契约测试。
NPU完整套件共145项，无 skip/xfail/deselect。

静态扫描清单 `hisparse-npu-upstream-inventory.json` 记录150个 HiSparse 命名或直接调用
四算子的函数，以及20个 `requires_hisparse_ops` 函数。这不是150个原始测试都已运行。
四个不归入本次四算子移植的 gated 函数为：

- `hisparse_rejects_union_bound_beyond_shared_memory`：CUDA 专用硬件容量条件；Ascend改验证workspace/union边界，并实际通过H32768/K8192。
- `hisparse_kv_update_writes_resident_cache`：backend KV update 集成，未执行原函数。
- `hisparse_row_dma_copies_discontiguous_spans_across_layers`：backend row DMA 集成，未执行原函数。
- `hisparse_mixed_batch_bf16_row_split`：依赖 FlashMLA attention/model metadata 的集成，未执行原函数。

同样不宣称其他 HiSparse 配置、connector、scheduler、attention 或模型测试已通过。

## 失败与修复

1. **生产 kernel 设备编译失败**：Ascend Triton 前端不支持 `break`，CPU interpreter
   不会暴露此问题。把 hash probing 和 request 连续行扫描改为有界条件 `while`，
   保留顺序、union 上限和全部比较语义。生产变更仅涉及这两处循环。
2. **测试输入准备失败**：NPU `arange(dtype=int16)` 不支持（161002）。改为在测试准备阶段
   构造CPU int16再传NPU，生产 kernel 无 staging/同步回退。
3. **负例断言的错误归因风险**：早期仅检查子进程非零退出，可能把上述输入准备失败误判
   为预期超限失败；该次“1 passed”不作为超限验证证据。最终负例必须同时满足：合法
   warmup成功、所有输入准备完成标记、子进程非零退出、没有成功完成标记，并明确收到
   CANN `507035` 及 `timeout or trap error`。子进程stdout/stderr/exitcode全部保留。
4. **错误消息的平台差异**：A3 的 `tl.device_assert` 降为trap，Python异常不包含源码断言
   字符串。最终按实际平台错误码和trap特征验证；数值、索引、LRU、stats、字节比较没有放宽。
5. **独立运行的CANN context**：测试映射若在任何NPU tensor分配前调用注册会得到107002。
   测试专用映射先显式 `torch.npu.set_device(npu:0)`；最终完整套件和stream用例通过。

重要批次位于工作区 `.artifacts/`：

| 目录 | 结果 |
| --- | --- |
| `hisparse-existing-runtime-20261102-145316` | 原26个invalidation全通过，22.62秒 |
| `hisparse-npu-ut-20261102-145701` | resolver首次设备编译失败：Break |
| `hisparse-npu-ut-20261102-145751` | 修复后resolver smoke通过 |
| `hisparse-npu-ut-20261102-145841` | 15个gather布局用例通过，16.10秒 |
| `hisparse-npu-ut-20261102-150143` | resolver76项通过，116.28秒 |
| `hisparse-npu-ut-20261102-150549` | 业务规模、联合回归等59项通过，55.75秒 |
| `hisparse-npu-ut-20261102-150752` | 早期负例1 passed；因准备阶段错误归因不足，不作为最终超限证据 |
| `hisparse-npu-ut-20261102-151123` | 6 passed / 1 failed：stream测试准备的int16 arange |
| `hisparse-npu-ut-20261102-151252` | 负例真实trap已发生，但自定义错误文字匹配不适用于A3 |
| `hisparse-npu-ut-20261102-151426` | 最终负例通过，独立stream前注册因context未选择失败 |
| **`hisparse-npu-ut-20261102-151736`** | **完整145 passed，21.58秒；最终有效NPU证据** |
| **`hisparse-ops-20261102-151820-all-cpu`** | **CPU140 passed / 8 deselected，20.08秒；临时容器已清理** |

## 复现

在130运行。先重新检查设备计算进程和容器可达性；已有容器后续可能变化。
runner 不创建/停止已有容器，不安装依赖，不开启 device-share，不使用 `wall`。
任务进程通过容器内外双层 `timeout` 限时，最大13分钟。

```bash
cd /mnt/share/h00954440/HiSparse-Ascend-opsWorkWithCodex/vllm-ascend

# 完整NPU套件，默认不筛掉任何测试
bash tools/run_hisparse_npu_tests.sh 8 vaws-weiguihua-20261030

# 可选聚焦运行，表达式作为一个参数传递
bash tools/run_hisparse_npu_tests.sh 8 vaws-weiguihua-20261030 'npu_resolver'
bash tools/run_hisparse_npu_tests.sh 8 vaws-weiguihua-20261030 'npu_gather'

# 无设备映射的CPU回归
bash tools/run_hisparse_ops_tests.sh cpu all
```

最终 NPU 内部命令为（精确转义版保存在该批次 `command.txt`）：

```bash
python3 -m pytest --confcutdir=tests/e2e/pull_request/one_card \
  tests/e2e/pull_request/one_card/test_hisparse_ops_npu.py \
  tests/e2e/pull_request/one_card/test_hisparse_invalidate_written_slots.py \
  -k '' -x -vv --tb=short \
  --basetemp=<artifact>/tmp --junitxml=<artifact>/results.xml
```

每批记录完整命令、环境版本、镜像、两个提交、未提交源码SHA-256、日志、JUnit XML和
运行前后NPU状态。测试使用的模块通过文件路径加载真实生产 `hisparse.py`，不复制或
模拟 kernel。源码hash比HEAD更重要，因为交付保持未提交。
AST、Bash语法、新增文件空白和120列检查已执行；Ruff不可用，未执行格式工具或完整CI。

最终结果已经验证这张A3上四算子的编译及上述执行语义；backend接入、完整runtime、
attention/model验证和性能优化仍是后续工作。
