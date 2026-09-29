# KV Block Trace 模块定位与验收

## 职责

记录并查询 KV block 生命周期、请求映射、搬运过程，以及选定观测窗口内的字节变化。输出是带来源、作用域和覆盖限制的事件，可供人工或其他工具消费。

这个功能的完成标准是：已声明支持的路径能够如实采集，相关事件能够关联检索，缺失证据不会被补造成事实。

## 边界

| 属于 trace 模块 | 由外部工具或调用者负责 |
| --- | --- |
| 分配代次、引用和复用记录 | 完整响应采集、复读率与模型质量评估 |
| Scheduler/Worker 映射与真实 slot 记录 | 标杆生成、teacher forcing、逐层数值比较 |
| 搬运来源、目的与 host 返回事件 | 修改业务写入、mask 干预、修复验证 |
| 选定 block/layer 的前后差分 | 自动阻断、缓存修复与最终根因裁定 |
| 事件完整性及关联依据 | 承诺每个故障都能定位或没有未观测污染 |

后续 backend 适配扩展事件覆盖。L1 状态机、语义校验或 checksum 可以消费/补充这些事件，但不作为当前 trace 模块必须内置的诊断算法。

## 本轮完善

- Linux 日志新增 host boot 与进程启动身份，防止跨启动/PID 重用混淆不同 writer。
- Reader 从同一进程的调度上下文确定 pool，从同一 host/boot 的分配和引用记录还原 dummy 开始时的关联上下文。
- 记录最近观测到的同进程搬运事件；跨新分配代次清除旧搬运关联。搬运上下文不是设备完成或正确性证明。
- 请求/epoch 查询保留相关 dummy begin、diff、end，以及关联所引用的原始事件。
- 使用 `access_relations` 单独承载派生关系，不给 dummy 的原始 `request_ids`、`pool_id`、`alloc_epoch` 补造值。
- 支持 host、DP/TP/PP rank 过滤。原始块查询仍可查看关联未知的访问。
- 有缺失序列、观测失败、作用域歧义或生命周期事件同时间戳时，关联保持 unknown。旧日志使用较弱的 `legacy_host_pid` 依据，并明确标注。

## Case 4 对照

该案例的关键观察是：传输完成后，空 batch 的 dummy 用 `slot=142` 写了 block 1 / offset 14，最后一层 latent KV 与 RoPE K 分别改变 512、64 个元素。

这些属于当前采集范围；本轮完善让它们能以请求或块代次检索，并显示支持关联的分配/引用/传输记录。复读响应、19-token prompt、抢占指标以及 mask 开关实验由外部流程提供，无需移入 trace。

普通 MLA、非 NZ、已采样 block/layer 的差分属于支持范围。未采样历史块、窗口开始前的变化、raw KV 数值对比、未接入 connector/runner 路径仍不承诺。单纯缺少这些能力不代表当前模块违约；错误声明覆盖或遗漏其承诺采集的事件才属于缺陷。

## 验收

1. 分配重用递增 epoch；合法前缀共享保持 epoch。
2. 请求/epoch 查询包含有关 dummy span，并可追到关系依据；相关请求不能被误当成同一请求别名。
3. 相同 block 号在不同 host、启动、engine、DP 或 pool 中不串线。
4. 原始事件不变；派生上下文标明 host-observed，不能声称 device epoch 已验证。
5. 窗口中有新传输、引用变化或再分配时，差分带上下文变化提示。
6. 日志有缺口时停止推断，保留 raw block 观察和 incomplete 信息。
7. 已有差分、错误传播及关闭 trace 的行为不因关联查询改变。

本轮关联逻辑主要位于离线 reader；运行时仅新增进程身份字段，无额外 tensor 采集。后续已完成 [Case 4 原环境实机验收](kv-block-trace-case4-validation-20260929.md)：131 上原镜像复读 15/20；开启采集后复读 4/10，4 次均捕获传输完成后空 batch dummy 的 offset 14 改写，且请求/epoch 查询完整保留。该结果覆盖原 DP2 图模式的采样路径，不代表长稳或性能已验证。

关联实现阶段检查结果（下列旧 workload 与后续 Case 4 实验分开记录）：

- `python -m pytest --confcutdir=tests/ut/debug tests/ut/debug -q`：70 passed，包含请求/epoch 保留 dummy、复用、共享、跨作用域隔离、缺失日志、同时间戳、窗口中重分配和重复读取查询输出。
- 修改的 Python 文件通过 Ruff 和格式检查；7 张 SVG 的放大、下载、锚点与窄屏检查通过。
- 131 的 Linux 进程身份探针通过：同进程 worker/transfer 一致、子进程不同、日志字段与探针一致。没有运行新的 NPU workload。
- 保存的 Qwen、PD、DeepSeek 实测日志共 152 / 2559 / 1205 条，经过新 reader 后原始字段不变，原有完整性结论分别保持 0 / 9 / 1 项问题。它们不是此次关联补充在原 Case 4 图模式上的新验证。

Case 4 实机暴露的 `cache.config.groups` 无 slots 查询问题已修复，新增回归后为 72 passed。16 个 writer 缺少 stop，完整性检查如实报 incomplete；不能把窗口证据充足表述为整份日志闭合。
