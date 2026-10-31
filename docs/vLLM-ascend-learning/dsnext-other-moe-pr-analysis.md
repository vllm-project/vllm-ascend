# DSNext 其他已合并 PR 的 MoE 升级分析

## 结论与核对范围

除了已经合入的 !18、!31、!51，确认还有 **5 个直接修改 MoE 实现的已合并 PR：!17、!54、!69、!70、!72**。另外，!52、!60、!73 修改 MoE 通信选择或相关缓存；!8、!58 涉及模型入口或 draft 专家配置。!24、!76 与 Engram/调度的兼容性有关，应按实际启用的功能评估。

当前对比目标：`integrate/dsnext-megamoe-vllm-cann`，HEAD `203fbd7d43268cbca4ecb6b2a8a8aa4f43e57839`，基线 `rfc/vllm_cann@0040652a66d31a2b277588ac90d7d37e92757bc5`。

来源：`xujiuxu9/vllm-ascend-dsnext`。通过远端 `ls-remote` 核对，`releases/v0.27.1rc` 仍为 `78758949e8163b022d47386d85e84028bf051669`，`ds-next` 仍为 `fb53d417a51f4277598120edf6ef6101cb2bb5c9`，与本地来源记录一致。扫描两个分支全部可达的带 `!编号` 合并提交，共确认 44 项，并按合并结果相对第一父提交的实际差异判断影响范围。每个 PR 的文件和来源提交保存在 [dsnext-merged-pr-inventory.json](../../dsnext-merged-pr-inventory.json)。

GitCode PR 页面及公开 API 未能读取，因此 PR 编号和合并状态依据 Git 合并记录。来源 main/RFC 的已取得最新提交中未发现额外带 `!编号` 的第一父合并记录；其他尚未取得的分支、被 squash/fast-forward 合并且没有 PR 编号的提交，以及服务端已关闭但没有合并的 PR，不能据此穷举。这是一份已确认记录的分析，不把未知状态当成已合并 PR。

## PR 对照

| PR | 合并提交 | MoE 相关内容 | 当前版本状态与处理 |
| --- | --- | --- | --- |
| [!8](https://gitcode.com/xujiuxu9/vllm-ascend-dsnext/pull/8) | `fb53d417a` | 早期 V4.1 MoE 类、模型注册和加载入口 | 当前已有完整 V4.1 模型目录；不回迁旧平铺模型文件。若关注特定字段或权重键，按当前模型单独验证。 |
| [!17](https://gitcode.com/xujiuxu9/vllm-ascend-dsnext/pull/17) | `da075d4a6` | V4.1 基础支持、视觉/hash 路由、force EPLB、DS FP8 scale 按 block size 展开 | 当前已有视觉/hash 路由、force EPLB 和按 block/group size 展开的 scale；主要能力已覆盖，不整包合入。保留测试参考。 |
| [!52](https://gitcode.com/xujiuxu9/vllm-ascend-dsnext/pull/52) | `72659cb16` | 缓存 A5 MoE 量化/激活，避免每步扫描 model.modules() | 当前通信选择已走硬件策略和显式 draft 量化参数，旧扫描函数已不存在；无需恢复旧缓存。 |
| [!54](https://gitcode.com/xujiuxu9/vllm-ascend-dsnext/pull/54) | `d292b4654` | DSpark target/draft 拓扑识别、通信实例按拓扑隔离、异构 draft 禁用旧 MegaMoE、纯动态 sqrtsoftplus 路由回退 | 当前有按专家数隔离的通信实例和 draft 量化检查，但不是源补丁的完整拓扑机制。需要按当前 FusedMoEConfig、上下文和算子能力重建缺失部分，见下文。 |
| [!58](https://gitcode.com/xujiuxu9/vllm-ascend-dsnext/pull/58) | `7a4ae74bd` | 修正 V4.1 名称与 DSpark num_experts_per_tok 配置字段 | 当前 `_normalize_deepseek_v4_dspark_draft` 已使用 `dspark_num_experts_per_tok` 并更新 model_arch_config；对应修复已覆盖。 |
| [!60](https://gitcode.com/xujiuxu9/vllm-ascend-dsnext/pull/60) | `8115b5102` | 防止旧量化/激活缓存受配置对象 ID 复用影响；同时有 DSpark、Engram、分组等改动 | 缓存路径已被当前硬件策略替换，不搬旧缓存。其他改动须按 DSpark/Engram 功能独立评估，不能将整个 PR 当成 MoE 补丁。 |
| [!69](https://gitcode.com/xujiuxu9/vllm-ascend-dsnext/pull/69) | `b7029879e` | 在 D 节点给异构 draft 建立 mc2_draft 通信组，让 draft MC2 不与 target 混用 | 当前没有 mc2_draft 或 resolve_mc2_group_for_topology。若启用不同专家拓扑的 DSpark，需要适配组选择、dispatcher 注入和 HCCL 生命周期。 |
| [!70](https://gitcode.com/xujiuxu9/vllm-ascend-dsnext/pull/70) | `763c89bc6` | 将 draft 独立通信组扩展到混合部署；缺组时报错；关闭共享多流时不再等待同流 event；另含大量 Engram 优化 | 共享同流免等待已有 `_wait_for_milestone` 对应机制；draft 组扩展依赖 !69，当前仍未具备。只提取该链路缺失的行为，保留 RoutedMoEMilestones。 |
| [!72](https://gitcode.com/xujiuxu9/vllm-ascend-dsnext/pull/72) | `206d3bb77` | A5 gate Cast+GEMM 融合、减少冗余 FP32 Cast、复用图像 token mask、视觉算子 scaling 参数修复 | 当前已有 DeviceOperator.compute_gate_logits/16cast32 和一次 scaling 传递。图像 mask 逐层计算仍有优化空间，但要适配现有 hash/vision 算子与 TP token 布局。 |
| [!73](https://gitcode.com/xujiuxu9/vllm-ascend-dsnext/pull/73) | `a5690af52` | private circle pool 下有条件跳过 DP all-reduce，避免不同 rank 选不同通信方式 | 当前没有该旧 private circle pool 补丁，不需要单独迁移。若引入等价部署模式，重新验证 skip 条件与现有 MegaMoE token 同步。 |
| [!24](https://gitcode.com/xujiuxu9/vllm-ascend-dsnext/pull/24) | `3935bf5f5` | 同步 Engram 与 elastic EP 的兼容性限制 | 属于功能组合约束，不能仅靠 MoE 文件合入；需按当前 Engram/elastic EP 实现评估。 |
| [!76](https://gitcode.com/xujiuxu9/vllm-ascend-dsnext/pull/76) | `779f109fc` | private pool 下拒绝 D 节点混入 prefill，避免 DP MoE 通信选择分歧 | 属于调度与 KV pool 行为；不作为独立 MoE 内核补丁迁移。 |

## 最需要适配：!54 → !69 → !70 的 DSpark 多拓扑链路

这条链路应在启用不同专家数量或 top-k 的 draft 时优先处理。源补丁依赖旧 `_select_a5_moe_comm_method`、旧拓扑 registry 和旧全局 forward context，直接 cherry-pick 会覆盖当前硬件策略。

1. **拓扑来源与通信实例键**：当前 `_moe_config_key` 只含 `num_experts` 和 `num_local_experts`。源 !54 使用专家数、top-k、EP 大小。不能把源 `(num_experts, experts_per_token, ep_size)` 元组直接传给当前要求 FusedMoEConfig 的 `get_moe_comm_method`。需确认 target/draft 实际 MoE config，并验证同专家数但不同 top-k 等组合不会共享尺寸不同的实例。
2. **draft 可用的融合路径**：当前检查 draft 量化是否支持 A5，而源补丁还排除了不同专家拓扑。新 CANN 的缓冲区是否真正支持多拓扑不能凭旧限制下结论；应同时验证上下文、专家数、top-k、容量和 CANN ABI，保证不使用 target 的 buffer/权重解释 draft。
3. **独立 MC2 group**：!69 最初只在 D 节点创建，!70 修正为在配置存在异构 draft 时创建，覆盖无 KV transfer 的混合部署。各 rank 创建条件和 new_group 顺序须一致；draft dispatcher 的 group name、rank、world size 必须来自同一组。
4. **恢复与销毁**：当前 dispatcher 和 MegaMoE 已有 HCCL 恢复机制。新的 draft group 需要接入 destroy/recreate/refresh，不能只改首次初始化。当前通信实例缓存也必须保持隔离。
5. **共享专家与事件**：继续使用 `RoutedMoEMilestones`，不要恢复源代码的 `FusedMoEEvents.before_routed_experts` 等旧成员。!70 的同流免等待目的已覆盖。
6. **无 hash 的 draft 路由**：源 !54 假设旧 hash 算子必须提供 input_ids/tid2eid，所以加了 Python 参考实现。当前 C++ tiling 把它们作为 optional input，并存在无 input_ids 的 non-hash 路径；不能据此认定当前缺少该功能。需要验证 correction bias 只影响专家选择、权重取自原始分数，以及 renormalize/scaling 的数值语义。

最低验证组合：target/draft 不同专家数、相同专家数但不同 top-k、不同量化；P、D、无 KV transfer；MC2/ALLGATHER 回退、dummy DP rank、图捕获/回放及 HCCL 重建。MRv1 与 MRv2 应分别验证，不能将旧 V1 测试结果外推到 V2。

## !72 的可选性能优化与数值检查

当前 A5 gate 已通过 `DeviceOperator.compute_gate_logits` 调用 `npu_fused_matmul(..., fused_op_type="16cast32")`。最近的共享专家融合适配也调用该现有接口，因此不需要再引入 `_gate_router_logits`、`is_950()` 分支或 layer.weight_t 缓存。

源补丁删除 decoder 层 `.float()` 的方式不能直接应用：当前 decoder 使用 `rms_norm_cast` 同时提供归一化结果和 FP32 路由输入，并支持 SP、非 A5 和多种 gate。优化前需确认哪些输入确实未被使用，再测精度和显存。

源码的图像 mask 缓存仅按 token 数检查命中。若在当前版本实现复用，需要明确一次 forward 的生命周期、sentinel 范围、padding、ALLGATHER/MC2/TP chunk 后的 token 顺序、不同层的输入布局，以及 ACLGraph 每次回放重新计算的要求。不能跨 step 仅因 token 数相同就复用上一次请求的 mask。

当前主要融合视觉路径使用自有 `moe_gating_top_k_hash`，不再是源 !72 操作的 `torch.ops.custom.npu_moe_gating_top_k`。当前代码已直接传递 routed_scaling_factor；Python fallback 在归一化后应用该系数。应补充文本动态、hash、视觉混合场景的参考数值比较，而不是搬用源补丁删除归一化的代码。

## 另一组实质缺口：Omni EPLB 提交链

来源分支在最后一个有编号合并提交后还存在以下 8 个 MoE/EPLB 提交。它们已在来源 branch tip 中，但 Git 日志没有对应 `!PR` 合并记录，不能编造 PR 编号：

| 提交 | 内容 |
| --- | --- |
| `9323200a0` | DeepSeek V4 的 Omni EPLB/OmniPlanner 接入 |
| `1b513624f` | 允许 W4A8MXFP 使用 Omni EPLB |
| `418881e3e` | 按每层实际专家数处理权重 |
| `3459fd399` | 从 Omni 管理的层中排除 draft MoE |
| `f6cc9c241` | 放置专家前同步计算流，避免异步推理读写冲突 |
| `04e4ef3a5` | 核对专家物理位置与已加载权重的指纹 |
| `cd1609fb2` | 指纹处理适配 NPU 私有格式 |
| `6ffbb214b` | 只对可 reduction 的专家权重计算指纹 |

当前版本没有 `OmniEplbUpdator`、`enable_omni_eplb` 或 `omni_config_file`；已有原生 EPLB 和 W4A8MXFP/NZ 相关能力，不等于已有 Omni 接入。

如要迁移，需要完整适配：

- 增加并固定 `omni_placement` 依赖与接口；区分模型初始化、权重加载、planner 初始化、warmup、forward、shutdown 的顺序。保留当前 model loader、RL reload 及 MRv1/MRv2 生命周期。
- 区分动态负载均衡与“权重拆成 per-expert list”两种语义。源 `convert_weight_to_list = dynamic_eplb and not enable_omni_eplb` 不能简单替代当前所有 dynamic_eplb 判断。Omni 使用 stacked 权重，而当前 D2D/NZ 路径有自己的布局要求。
- 对齐当前 `make_eplb_placement_config`、冗余专家槽位分配、log2phy、expert_map 和热度统计；按每层实际专家数处理 target/draft。源通过 local_num_experts 相同来筛选 target 层也有边界：相同专家数的 draft 不应仅据此被误纳入。
- 验证 Omni 移动专家后 CANN 缓冲区、独立权重副本和路由映射是否仍一致。共享专家融合当前排除 dynamic EPLB；迁移初期保持该约束，不能把“无需转换成 list”误当成“可以融合”。
- 同步计算流的源修复会引入等待。需要在 NPU 上验证异步调度、图回放和精度，再评估性能，不将去掉同步作为默认优化。
- 不将私有 NZ 权重直接转 view 后 reduction。源最终指纹方案跳过不支持 reduction 的权重而使用可处理的 scale；需要明确指纹的覆盖范围，它不是逐元素相等的证明。
- 源 `os.getenv("DYNAMIC_EPLB")`、`os.getenv("EXPERT_MAP_RECORD")` 和 `VLLM_ASCEND_OMNI_EPLB_VERIFY_INTERVAL` 的读取方式需遵循本仓规范，统一放入 `envs.py` 或转换成配置字段，并记录默认值和语义。

## 建议顺序

1. 启用异构 DSpark 时，先将 !54、!69、!70 的缺失行为按当前接口组成一个独立补丁，并验证通信及精度。
2. 若只运行普通 target/MTP，先完成现有 MegaMoE 共享专家融合的 NPU 验证；不要为未启用的 draft 引入额外通信组。
3. !72 的图像 mask 复用与冗余 FP32 输入优化作为独立性能任务，以 profiling 和路由精度为验收依据。
4. 只有需要 Omni EPLB 时才引入上述完整提交链；!17、!52、!58、!60 中已经被新架构覆盖的旧实现不回迁。
5. !73、!76 仅在引入 private pool 对应部署模式时评估；Engram/调度/缓存升级与 MoE 升级分开验证。

本次仅分析 Git 差异和当前接口，未修改生产代码，未运行或声称通过 NPU 推理、精度、性能测试，也未推送或创建 PR。
