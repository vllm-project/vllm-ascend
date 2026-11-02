# HiSparse 四个 Triton 算子：CPU 复现与 CUDA 契约
>
> 提交前复验（2026-11-02）：统一导入入口及 Ruff 整理后，A3 145 passed（164.16s），
> CPU 140 passed、8 个 NPU 用例 deselected（23.03s）。最终日志、XML、源码哈希及
> 检查边界见 [validation evidence](hisparse-validation/2026-11-02/README.md)。下文保留各阶段执行记录。
>
> 后续NPU阶段已完成：见 [A3执行与复现记录](hisparse-npu-reproduction.md)。
> 最终完整NPU套件145 passed，CPU回归140 passed；下文保留最初CPU阶段的范围和历史结果。

## 范围和基线

本阶段在 130 的 `/mnt/share/h00954440/HiSparse-Ascend-opsWorkWithCodex/vllm-ascend`
开发。两个 worktree 的分支均为 `HiSparse-Ascend-opsWorkWithCodex`。

| 仓库 | 锁定完整提交 |
| --- | --- |
| vLLM（只读参考） | `d5301a4f5487db1d7e8d87daf7828225b95b1774` |
| vLLM-Ascend（未提交改动） | `cae7ff0290dd6c6155f71d7223e0fd4aff95c9e9` |

只增加算子、测试和复现工具；不注册 backend，不修改 vLLM，不 commit、不 push。
参考是冻结版本的 `vllm/csrc/hisparse_kernels.cu` 和
`vllm/csrc/torch_bindings.cpp:1055-1108`，没有使用 Ascend PR #19109。
Ascend 基线声明的 vLLM 验证提交为 `544f182d9f0c9bfe8bd7645525951d5e3692a542`，
它是本次 vLLM 基线的祖先，相差 638 个提交。CPU 算子通过不能证明这组版本运行兼容。

## 实现入口和约束

生产文件：`vllm_ascend/ops/triton/hisparse.py`。

- `HiSparseTritonCacheOps` 提供 CUDA schema 对应的四个方法，原 invalidation
  六参数独立入口保留，四个方法均原地写输出、返回 `None`。
- `HiSparseResolverWorkspace(device, max_rows, num_states, top_k, hot_size, max_union_rows)`
  显式预分配。每个 index group 独占 workspace；同一有序 stream 复用，跨 stream 并发
  必须使用不同 workspace。workspace 容量检查只读 shape 等 Host 元数据。
- 每个 request 连续行段只有 leader program 写 state。哈希表、首次引用位置、union 顺序、
  hit/evict 分区和统计 scratch 存放在有界全局 workspace；无 `[top_k, hot_size]` 广播。
  首版保留顺序循环，目标是可审查的语义正确性，没有吞吐或 A3 可用性承诺。
- `HiSparseHostMapping` 是注入协议：`device` 和 `address_of(host_cache)`。backend 必须
  校验整个视图范围、映射仍有效，返回**视图起始位置的设备可见地址**，并持有映射到
  stream 完成。gather 拒绝缺失/设备不符/零地址的 mapping。
- 生产 gather 不读取普通 CPU `data_ptr()`。代码没有 `.cpu()`、`.item()`、同步 staging
  或自动回退。Host 注册、注销、硬件能力探测及图模式映射生命周期留给 backend 阶段。
- 公开入口要求 NPU，使用目标设备当前 stream。CPU UT 显式调用 `_resolve`/`_gather`
  来绕过这个平台检查，执行同一生产 kernel；测试专用 `CpuMapping` 仅返回 CPU 真实地址。
  这不证明 Host 内存可被 NPU 直接访问，也未验证 NPU-only wrapper 的设备环境。

## CUDA 契约对照

| CUDA 行为 | Triton 实现和 CPU 断言 |
| --- | --- |
| Resolver 全部参数、可选输出及默认值 | 方法与 schema 同名同序；64 种输出组合；单独覆盖可选输入 |
| 直接 Host 行号 / request-relative token | 整数页表转换，source block 0 无效，负值和越界处理 |
| Request/state 重排与 graph padding | 用 request 查 hot/source table，用 state 查 ownership/resident；按 `num_valid_rows` 裁剪 |
| Resident 优先 | resident 不计 Host hit/miss，resolved 输出仍为 Host 行号；完全 resident 保持 ownership/LRU |
| 多 query 行 union | 按 `(row, column)` 首次引用顺序去重，所有旧 hit 在淘汰前受保护 |
| Miss 分配与 compact plan | miss 依次取最旧可淘汰槽，plan 从 leader 行起连续打包；共享 miss 只有首次引用 mask=1 |
| LRU | 剩余可淘汰槽 → 新载入槽 → 本轮 hit；各段稳定顺序 |
| 统计 | `uint64[2]` 累计 hit/miss，后续共享引用计 hit，resident 不统计；整数相加保留 uint64 回绕 |
| 容量超限 | 插入有界 order workspace 前 `tl.device_assert`；launch 强制 `debug=True`，不截断 |
| 损坏 LRU 的防越界分支 | 非法 eviction slot 不写 ownership，plan 目标和输出为 -1，仍保留该条 compact count |
| Strided metadata | 支持 global/hot/source/resident table 行 stride、attention 行 stride、valid-count stride |
| Gather opaque payload | uint8 指针逐字节搬运，保留 NaN 等原始位模式，允许 Host/Hot dtype 不同但字节宽一致 |
| `gather_plan` | 2D staging / 3D paged；padding 和 miss mask 控制 copy；attention 独立于 mask 生成 |
| `gather_compact` | 只处理 `clamp(miss_counts, 0, top_k)` 前缀；只接受 3D paged hot cache |
| 两个 gather 的非法行 | 负源/非法目标不写；非负源超 Host 范围且目标有效时清零 |
| 真实物理布局 | page stride、层间空隙、Host/Hot 子视图、非对齐目标地址；比较整个 backing storage |
| Invalidation | 保留六参数 ABI 和单写入者实现；原 18 CPU、8 NPU 用例保留 |

默认沿用 CUDA 合法输入前提：同 request 行连续；每行有效 top-k 唯一；活动 request
不共享 state ownership；ownership 中有效 Host 行唯一；LRU 是槽位排列。跨行重复合法。
不同 payload 对同一个 gather 目标的冲突写不是合法 plan。
不照搬 CUDA shared-memory 容量限制。A3 对 device assert 的编译/报错行为尚未验证。

## 上游 UT 移植清单

来源均为冻结版本的 `vllm/tests/v1/attention/test_sparse_mla_backends.py`。
该模块的平台 gate 不能直接用来证明 Ascend 通过；这里提取算子场景，替换 runtime、
`torch.ops._C_cache_ops` 和设备入口。下表不宣称原完整 runtime 测试已通过。

| 原用例名称（省略 `test_`） | 本地用例 / 保留与改动 |
| --- | --- |
| `hisparse_uses_graph_stable_request_state_mapping` | `resolver_cuda_contract[reorder]`、`resolver_reads_current_request_ids`：保留 request/state 间接映射和重复使用 scratch，未验证 graph |
| `hisparse_maps_speculative_rows_through_request_state` | `resolver_cuda_contract`：连续 query rows、非恒等 state 映射，缩小 K=4 |
| `hisparse_speculative_rows_resolve_host_misses_consistently` | `resolver_cuda_contract[shared]` 和联合 regression：保留跨行重叠与一致目标；CPU 缩小行数/K，长 MTP 待硬件 |
| `hisparse_speculative_rows_count_a_shared_load_once` | `resolver_speculative_rows_count_a_shared_load_once`：保留两行相同 top-k、K=128、block=64、hot=256、统计 `[128,128]`；另断言 plan count `[128,0]` 和首次 mask |
| `hisparse_resolver_reads_current_request_ids` | 同名 `resolver_...`：四行同 request → 一行有效加 padding；保留新输入生效断言，去掉 CUDA sleep/异步竞态设置 |
| `hisparse_union_table_scales_with_union_bound` / `hisparse_expanded_hash_keeps_union_bound` | `resolver_union_bound_is_error`、`optional_translation_inputs[hash_collision]`：保留超限报错、冲突探测和相邻 state 保护；K=8192 未执行 |
| CUDA shared-memory 容量拒绝用例 | 不移植硬件限制；本实现使用全局 scratch，分别检查 workspace 容量和 union 界限 |
| `hisparse_resident_rows_bypass_hot_lru` | `resolver_cuda_contract[all_resident]`：保留 resident 索引、valid count、ownership/LRU 不变；source 与 resident 的 block size 可分别设置 |
| `hisparse_swap_in_preserves_rows_across_eviction` | `regression_swap_in_preserves_rows_across_eviction`：保留 4 request、5 步 `(0,0,1,2,0)`、末列 -1、256/528/656 字节和累计统计断言；K=4、hot=8、block=2；独立 K=128 元数据用例另测 |
| `hisparse_multi_step_swaps_match_independent` | `resolver_multistep_stable_eviction`：五步，K=1/8/32/128，逐项比较独立 Python reference；未保留完整 runtime |
| `hisparse_multi_step_writes_request_major_output` | 上述多步及 `regression_resolve_gather_replay_and_rewrite`：保留按 query row 的 compact 前缀及未写尾部哨兵 |
| `hisparse_remaps_strided_hma_rows_for_attention` | `regression_remaps_strided_hma_rows_for_attention`：保留 block=4、page stride=3 倍、非零 offset、attention 重映射公式；字节宽缩为16，明确访问第二物理 page |
| `hisparse_gather_zeroes_unaligned_destination` | `gather_zeroes_unaligned_destination`：原样保留 16-byte rows、33-byte storage、stride17、源1/目标1、guard=7 与清零断言 |
| `hisparse_gather_rejects_short_attention_block_stride` | 同名 `gather_...`：保留 block2、stride1 拒绝；替换为本适配层 ValueError |
| `hisparse_newest_write_and_recycled_slot_invalidation` | `regression_resolve_gather_replay_and_rewrite`：提取 resolver→gather→多组 invalidation→再 resolve/gather，断言新 payload；未测试 runtime append/allocator |
| `hisparse_invalidation_covers_every_index_group` | 现有同名用例：原表、状态映射、重写和未列出表保持不变的断言保留，真实生产 invalidation kernel |
| `hisparse_gather_prefill_cache_prefers_resident_rows` | `gather_bytes_layout_and_guards` staging 场景：mask=0 保留既有字节，padding 不搬运；未验证 prefill runtime、resident staging 分配或 attention |

另外增加独立边界用例：64 种 resolver 可选输出、非对齐 cache/Host 子视图、16/64/256/528/656
字节行宽、count 负数及超 K、非法源/目标、int64 slot、超过 `2**24` 的合法 Host 行号、
超过 `2**53` 的累计统计、uint64 回绕、空行/空 top-k、metadata 容量检查。
Reference 文件 `tests/e2e/pull_request/one_card/hisparse_reference.py` 使用字典、集合和
稳定列表计算所有预期值，不调用生产 kernel；gather 按整个 backing storage 精确比较。

## 复现命令

在 130 上执行。CPU runner 仅使用已有镜像，不安装依赖；无 NPU 设备、driver 挂载、
端口或网络。最多 720 秒测试、900 秒容器生命周期，退出 trap 和 systemd timer 双重清理。
只操作本次 `codex-h00954440-...-hisparse-ops-...` 容器。

```bash
cd /mnt/share/h00954440/HiSparse-Ascend-opsWorkWithCodex/vllm-ascend
bash tools/run_hisparse_ops_tests.sh cpu all
bash tools/run_hisparse_ops_tests.sh cpu resolver
bash tools/run_hisparse_ops_tests.sh cpu gather
bash tools/run_hisparse_ops_tests.sh cpu regression
bash tools/run_hisparse_ops_tests.sh cpu invalidation

# 原 invalidation 单独复现入口仍然保留
bash tools/run_hisparse_invalidate_tests.sh cpu
```

每次只需选择一个入口。runner 创建 `--system-site-packages --without-pip` 的临时 venv，
真实命令写入 artifact `command.txt`，其结构为：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 TRITON_INTERPRET=1 TRITON_DEBUG=1 \
  <artifact>/venv/bin/python -m pytest \
  --confcutdir=tests/e2e/pull_request/one_card \
  tests/e2e/pull_request/one_card/test_hisparse_ops.py \
  tests/e2e/pull_request/one_card/test_hisparse_invalidate_written_slots.py \
  -k 'not npu_' -vv --tb=short --junitxml=<artifact>/results.xml
```

测试 fixture 还显式设置解释器 `interpreter_builder.options.debug=True` 并在结束时恢复。
原因是本镜像的 GridExecutor 过滤了 launch `debug=True`，`TRITON_DEBUG=1` 也未传给
InterpreterOptions；不设置它会漏掉真正的 `tl.device_assert`。这只启用解释器本身的断言，
没有替换 kernel、assert 实现或预期结果。生产 launch 则始终传 `debug=True`。

固定镜像：

```text
sha256:8725ad72b9db273848f13894bad0e547a67567cccb2f8692b7e3588b2e959982
harbor.computing.lab.rnd.huawei.com/hisecr/vllm-ascend@sha256:eb30579f3fff978b200c52560ded96dad03751854afe0e7110f02a0c43e474fb
Python 3.12.13 (/usr/local/python3.12.13/bin/python3)
torch 2.10.0+cpu; torch-npu 2.10.0.post7; triton-ascend 3.6.0
pytest 8.4.2; numpy 1.26.4
```

## 实际执行、失败及修复

Artifact 根目录：`/mnt/share/h00954440/HiSparse-Ascend-opsWorkWithCodex/.artifacts/`。
所有执行均无设备映射，scope 记录为 `devices=[] privileged=false network=none`。

| 2026-11-02 批次目录 | 实际结果 / 解释 |
| --- | --- |
| `hisparse-ops-20261102-141653-all-cpu` | 122 passed，1 failed，8 deselected；union 超限未报错，未判为通过 |
| `hisparse-ops-20261102-141757-resolver-cpu` | 85 passed，1 failed，19 deselected；显式环境变量仍不能启用该解释器的断言 |
| `hisparse-ops-20261102-142110-all-cpu` | 46 passed，92 failed，8 deselected；启用解释器断言后，哈希 uint32 乘法触发溢出检查；诊断后改用 uint64 乘法，统计相加显式保留 uint64 回绕 |
| `hisparse-ops-20261102-142228-all-cpu` | 138 passed，8 deselected，0 skip/xfail；pytest 20.82 秒，总过程25秒；task 容器已删除 |
| `hisparse-ops-20261102-142810-all-cpu`（最终） | **140 passed，8 deselected，0 skip/xfail**；pytest 21.19 秒，总过程25秒；含新增空输入测试，task 容器和 timer 已清理 |

最终140个通过项由新增四算子 CPU 套件122项和原 invalidation CPU18项组成。
`test_hisparse_invalidation_covers_every_index_group` 在本轮实际执行并通过。
前述失败日志和 XML 均保留，不覆盖。
Triton 解释器存在 NumPy scalar conversion 弃用警告；没有过滤失败或放宽比较。

每批保留：`image.txt`、`ascend-head.txt`、`vllm-head.txt`、Git status、
`source-sha256.txt`（生产/UT/reference/runner/CUDA/schema）、`command.txt`、
`container-scope.txt`、完整 `pytest.log`、`results.xml`、`timing.txt`、`cleanup.log`。
复现时应先对 SHA-256，不能仅凭相同 HEAD 认定未提交源码相同。

## 静态检查和待验证项

- 实际执行 Python AST 解析、生产模块导入、Bash `-n`、新增文件
  `git diff --no-index --check /dev/null FILE`，并检查 Python 行宽不超过120。
- 镜像和当前本机没有可用 Ruff；**未执行 Ruff、格式工具或完整仓库 CI**。
  AST/空白检查不等同于这些检查通过。
- 原 invalidation 的18个 CPU 用例纳入联合回归，8个 NPU 用例按名字显式 deselect。
  它们覆盖 NPU wrapper 校验、graph replay、当前 stream/event；本轮都未执行。
- Resolver/gather 的 A3 编译、执行、断言报错、图模式、多 stream 顺序、非对齐字节写、
  Host 可见地址与 payload 搬运均待验证。CPU 解释器不证明设备编译可用。
- 业务规模 K=2048/8192、较大 request/union、长 MTP、H=32768、workspace 开销和
  热路径性能列为硬件阶段，不混入 CPU 通过结论。现有 invalidation width=2048 已 CPU 测试，
  这与 resolver 的 top-k=2048 是不同覆盖项。
- Backend 后续提供 Host 注册/映射、视图范围和生命周期保证、index-group workspace
  分配及 stream 所有权，再接 `create_cache_ops(runtime)`。目前不具备模型端到端交付。

所有改动保留本地未提交；vLLM 参考工作树应保持干净。
