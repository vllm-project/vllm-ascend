# M3.1 BF16 LoRA 本轮验证报告

本轮按用户确认的口径验收：**独立 BF16 存储/舍入边界参考通过，服务混批原始 token ID 严格一致；merged 差异单列。**
结论限定于 A3、真实权重减层模型、eager 和下表配置，不等同于 60 层完整模型或 A5 已验收。
旧 [VALIDATION.md](VALIDATION.md) 保留为前一轮 Q/K/V/O 合同结果。

后续基座 checkpoint 重载修复及独立冷启动对照见 [重载验证报告](VALIDATION-reload-bf16.md)。

## 配套和资产

- vLLM main 固定 `c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15`，没有修改上游源码；Ascend 基线为 `87f5f67bdda60b5bf6c6785909730d0f9bf9a43f`。
- 硬件为 A3 / Ascend 910_9382。使用 uv 管理的 Python 3.12.13 环境，保留现有 torch 2.10.0、torch_npu 2.10.0.post7 和 Transformers 配套。
- 原始 checkpoint 是 `MiniMaxM3SparseForConditionalGeneration`，60 层、59 个 safetensors，共约 795.51 GiB。
  验证模型取其中 4 层真实张量，保留 BF16 值和维度；960 个张量、23,485,394,944 字节。
- 前 3 层覆盖普通 attention/MLP，第 4 层覆盖稀疏 attention、indexer、共享及 128 个路由专家。
  hidden size 6144、Q heads 64、KV heads 4，共 414 个逻辑 LoRA 目标。
- 同时验证 VL 包装的 `language_model_only=True` 和文本 `MiniMaxM3SparseForCausalLM` 入口。
- 没有训练 adapter。测试生成确定性的标准二维 A/B，磁盘 FP32、执行 BF16，统一 rank，默认 rank8/alpha16。
  zero/A/B 使用独立资产；rank16、rank128、alpha24 另建资产，不覆盖已完成实验。

## 支持和修改

| 范围 | 本轮实现/证据 |
| --- | --- |
| Q/K/V/O | 保留原三目标包装兼容；实际加载、计算、混批回归通过 |
| index_q/index_k | 五段融合 QKV 映射；index_q 跟随 KV 分片/复制，index_k 复制；dense attention 仍只有三段，拒绝非空 indexer 权重 |
| 普通 MLP gate/up/down | 复用已有合并列/行线性层；逐目标及组合实测 |
| 共享专家 gate/up/down | 复用普通线性 LoRA，逐目标及组合实测；不等于 shared-outer 格式 |
| 路由专家 w1/w2/w3 | 复用现有 FusedMoE、上游 packing/缩放/EP 分片和 Ascend 低秩算子；没有新增 NPU 算子 |
| 部分专家/部分投影 | M3 专用 manager 将完整缺失目标补成零增量；半对 A/B 拒绝；无所属目标的 EP rank 保持空 adapter |
| 跨 DP 专家 | 每步同步 adapter 并集并统一 GPU 槽位；空闲 rank 保留实际专家 adapter，不用 warmup adapter 覆盖 |
| DP AllGather | 使用现有 input-id gather 对齐逐 token adapter 索引，padding 仍为 inactive；修复远端 token 错用本地 adapter 索引 |

M3 使用上游 manager 子类和模型声明的 worker-manager 扩展，其他模型保留原创建路径。
通用 AllGather 修改只在存在 LoRA context 且发生 DP/PCP/SP gather 时生效，原因是该通信层实际改变了 hidden-state 行布局。
上游 mixin 当前硬编码 worker-manager 类型，因此 Ascend runner 需要一个模型可选的创建分支。

## 实机结果

机器可读统计及源码/原始报告 SHA256 见 [validation-full-bf16-summary.json](validation-full-bf16-summary.json)。
复跑入口与完整命令见 [BF16_FULL_SCOPE.md](BF16_FULL_SCOPE.md)；原始 JSON、日志和数值诊断保留在测试环境的本轮 run 目录。

| 验证 | 结果和覆盖 |
| --- | --- |
| 最终 CPU 回归 | **1516 passed，1 skipped**；覆盖 LoRA、M3、MoE prepare/finalize 和相关 v2 runner/KV-cache 用例 |
| 仓库门禁 | `bash format.sh ci` 通过，包含 Markdown、代码格式及 staged secret scan |
| 增强独立数值检查 | **54/54** 加载/层级检查通过；BF16 超差元素为 0，加载切片及非目标片段精确一致 |
| 单目标检查 | 之前独立运行的 Q/K/V/O、index_q/k、普通/共享 MLP gate/up/down 共 **36/36**；这批结果不混入增强 54 项 |
| TP1/4/8 | 增强检查覆盖核心 profile、attention/indexer 与 all；TP8 实际进入 KV/index_q 复制分支 |
| TP2、TP2+EP2 | 先前各 **15/15** 加载/数值检查通过，另有 TP2 indexer-b merged 严格一致 |
| TP4+EP4 部分专家 | **3/3**；包含无 adapter 目标的 EP rank，不把空 adapter 误判为加载失败 |
| rank/alpha | rank16/all、rank8/alpha24/all、rank128/attention-index 各 **3/3** 增强检查；rank128 其他模块未验证 |
| 文本入口 | all profile 的 zero/A/B **3/3** 增强检查；此前核心 5 profile **15/15** |
| LLM 混批 | 3/17/129 token prompt、8 token decode、5 种批序；单独/混批一致，观察到真实 base/A/B 同批，C 复用槽位后 A 恢复 |
| indexer 长上下文 | **4097 token**；超过 16×128 选块容量，A/B 均改变 query 和实际 top-k，A 恢复且无 hook 控制一致 |
| HTTP TP1+DP2，无 EP | **36/36** 混批请求原始 token 严格一致，观察到真实 base/A/B 同批 |
| HTTP TP1+DP2+EP2 | **36/36**，真实混批证据通过 |
| HTTP TP2+DP2+EP4 | **36/36**，真实混批证据通过；最终源码重新启动服务复测 |
| HTTP 生命周期 | 额外 **12/12** base/B/C 请求一致；A 切回、卸载/重载后 token 一致 |
| 独立 merged | **13 PASS / 2 FAIL**，原始失败保留，不改容差、不把失败改成通过 |

独立数值参考读取磁盘 A/B，按实际 BF16 或 FP32 shrink 存储边界独立计算，再与 NPU primitive 比较。
`atol=rtol=1/64` 不变，理想 FP64 差异另外统计；所有非零 case 的最大增量信号至少达到容差的 10 倍。
专家检查显式覆盖全部 128 个专家，不依赖短 prompt 恰好路由到某个专家。
这些标准低秩诊断资产用于验证实现，不代表训练质量或业务准确率。

## merged 两项差异

五个核心 profile（indexer、MLP、共享专家、路由专家、all）各跑 zero/A/B：MLP-b 和 all-a 严格 token 不一致，其余 13 项一致。
例如 MLP-b 在线输出 `[7368, 150796, 33299]`，merged 输出 `[7368, 150796, 159294]`。
未修改的 `87f5f67` 也复现相同 MLP-b 差异，不能归为本轮新增回归。

合并端关闭 LoRA，独立读取原 checkpoint 和 A/B，计算
`BF16(FP32(W) + FP32(B_eff) @ FP32(A_eff))`，然后使用常规 loader；不复用在线模型输出作为参考。
捕获第一层相同输入后，73,728 个元素的在线/merged 输出各自对独立 BF16 参考最大差均为 0.0078125，超差元素均为 0；
两个独立参考本身不逐位相等，最大差 0.03125，与实测在线/merged 最大差一致。
后续层及离散选 token 可放大该差异。因此只按用户确认的 BF16 数值和服务一致性口径签收，不声明 merged 逐位等价。

## 已定位并修复的故障

| 修改前触发 | 证据 | 修复后 |
| --- | --- | --- |
| indexer adapter 不在合法模块映射 | `baseline-indexer-spawn` | 五段映射、分片与公开加载通过 |
| 只有专家 w1 的标准 adapter | `partial-expert-001` 的上游 w2 断言 | 完整缺失投影补零；w1/w2/w3 单项均通过 |
| EP rank 不拥有任何目标专家 | `public-ep4-subset-001` 的空集合 StopIteration | 空 adapter 正确保留，subset 回归通过 |
| DP 收到其他 rank 的专家 token | `http-joint-001` token 不一致 | 全局槽位/缓存同步后 TP2+DP2+EP4 通过 |
| DP 无 EP 的 AllGather 改变 token 行布局 | `http-dp2-noep-002` 仅 2/12 一致 | adapter 元数据同序 gather 后 36/36 一致 |
| VL 非 LoRA 的旧 dummy 配置 | `ut-001` 失败，基线同用例通过 | 以实际语言子模型能力传播，原 UT 未削弱 |

## 限制与审查

- 完整 60 层约 795.51 GiB BF16 权重需要整机资源；本轮没有 16 张可独占卡，未启动完整模型验收。A5 没有可用设备。
- 主闭环为 eager，最大短上下文配置 256；长上下文专项 4097。未签收 ACLGraph、128k/bs16 容量、PP/PCP、dynamic EPLB/FusedMC2、性能。
  通用 AllGather 修复的其他模型实机覆盖也未执行，不能由 M3 结果外推。
- 只声明标准二维、统一 rank 的语言侧 LoRA。3D/混合格式、异构 rank pattern、视觉、router、embedding、LM head、MTP 未验证。
- M3 明确拒绝 `fully_sharded_loras` 和 `enable_moe_shared_loras`；shared-outer 的因子专家轴为 1，与现有 NPU adapter×expert 索引不兼容。
- 跨 DP 路由专家明确拒绝 pin；本地 LRU pin 不能保证全局槽位重排后继续驻留。单 DP 沿用上游 pin。
- DP 的 `max_loras` 必须覆盖一个执行步全部 DP rank 的 adapter 并集；身份冲突/容量超限在槽位修改前明确报错。

按四个视角复核：

| 视角 | 具体位置与证据/边界 |
| --- | --- |
| 安全 | `vllm_ascend/lora/minimax_m3.py:92` 检查同 ID 的路径/名称/格式冲突；磁盘格式与合法目标继续由上游 loader 校验；测试服务仅 loopback，公开报告不含凭证/权重/激活 |
| 性能/资源 | `minimax_m3.py:72` 只对 DP 且存在专家 LoRA 的模型开启同步；`:104` 检查全局槽位预算；每步多一次 DP 对象通信及必要的 token 索引 gather，延迟未测，不宣称性能收益 |
| 边界 | `minimax_m3.py:18` 空所属专家、`:36` 半对 A/B；`utils.py:195` dense 非法 indexer；相关 UT、TP8 复制及真实 EP4 subset 共同覆盖 |
| 时序 | `worker/v2/model_runner.py:891` 实际/空闲 DP 步均同步，`:1343` 保留实际缓存；`lora/fused_moe.py:45` 清理逐次路由状态；HTTP 同批、槽位复用及 unload/reload 实测通过 |

复核同时尝试推翻缺陷假设：MLP-b 在旧提交亦失败，故撤销“本次适配造成 merged 差异”的归因；
DP 无 EP 仍实际 gather 远端 token，因此不能用“没有 EP 就无需 adapter 元数据对齐”排除故障。
未验证的配置保持未验证，不把静态检查、作者报告或 CPU UT 写成 NPU 通过。
