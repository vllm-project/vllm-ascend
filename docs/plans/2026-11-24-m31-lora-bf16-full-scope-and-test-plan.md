# MiniMax-M3.1-Flash BF16 LoRA 完整范围与测试方案

> 执行说明：本文件保留开发开始前的范围盘点与验收设计，状态列对应当时基线，不能作为通过报告。
> 实施结果见[本轮报告](../../tests/e2e/models/minimax_m31_lora/VALIDATION-full-bf16.md)。

2026-11-24 用户补充确认：以独立 BF16 数值参考和服务 token 一致性验收，merged 舍入差异单列，
验证后推送。下文原始严格 token 检查继续执行和保留 FAIL，不改成相似度通过。

**目标：** 在 BF16 推理场景覆盖 M3.1 语言模型的注意力、indexer、普通 MLP、共享专家和路由专家，并保留 merged 正确性、多 adapter 服务正确性两条独立验收线。

**架构：** 参考 vLLM 的模型专用 LoRA 层，复用已有线性/MoE 加载、分片、生命周期管理；Ascend 只补实际缺失的模块映射、模型专用包装器和设备执行适配。模型支持、adapter 磁盘格式和运行配置分别记录，不能仅凭 `SupportsLoRA` 声明推断全部可用。

**技术栈：** 配套 vLLM、vllm-ascend、PyTorch/torch-npu、标准低秩 A/B、safetensors、独立 CPU 参考、真实 A3/A5 NPU。

## 1. 版本与证据边界

| 对象 | 核查版本 | 用途 |
| --- | --- | --- |
| 配套 vLLM | `c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15` | 生产依赖基线，继续保持主仓配套 commit |
| 个人 Ascend 分支 | `m31-lora`，`87f5f67bdda60b5bf6c6785909730d0f9bf9a43f` | 当前代码与既有验证基线 |
| 上游参考 PR | [vLLM #63323](https://github.com/vllm-project/vllm/pull/63323)，head `4917985161655ab0ca549f8b6e07ead88338d078` | 五段 indexer LoRA、dense 兼容及测试思路；核查时仍为 Draft |
| 参考验证流程 | MindSpeed-Agent_9400 `d6b369ed8298326ec01119f73a5adbd948651448`，LoRA skill v1.0.2 | P0–P9 单请求流程及证据管理 |

既有本地报告见 [VALIDATION](../../tests/e2e/models/minimax_m31_lora/VALIDATION.md)。68 项通过只覆盖真实前 4 层、Q/K/V/O 合成 adapter、A3 单卡、BF16/eager、单请求、单驻留 adapter、TP/DP/PP=1。indexer LoRA、MLP LoRA、专家 LoRA、A5、多卡和混批没有这批证据。

上游 PR 的作者报告了 GPU 小模型验证；其真实 checkpoint、TP>1 和路由专家 adapter 未做端到端验证。作者结果不计入本项目 NPU 通过数。配套 commit 也没有该 PR 新增的 M3 专用类，当前不能直接 import 它们。

## 2. 范围定义与第一性原理

已验证事实是当前资产仅含 Q/K/V/O；“继承通用包装器就自动支持所有 M3 投影”仍需证明。不可变约束是 `ΔY = (X Aᵀ) Bᵀ × scale` 的张量语义、真实物理布局、并行分片及路由关系。将所有投影当作相同宽度、将共享专家等同共享因子，均不成立。

最小完整方案先建立每个磁盘 A/B 到运行 slice/expert 的清单，再逐类验证实际增量，最后组合验收。最快的证伪实验是使用真实层构造器和现有 `from_layer`，检查 wrapper 选择、加载后的 buffer 与非零输出；不得只检查配置字符串。

本次用户已明确要求纳入 indexer、MLP 和专家，主验收范围如下：

- 注意力：dense/sparse 层的 Q、K、V、O。
- Indexer：稀疏层的 index_q、index_k，含五段融合及其 TP 复制规则。
- 普通 MLP：gate、up、down。
- 共享专家：shared_experts 的 gate、up、down。
- 路由专家：每个专家的 w1/gate、w3/up、w2/down；完整及部分目标分别列账。
- 上述模块的组合 adapter、zero/base 对照、加载/卸载/重载、串行切换、混批和 TP/DP/EP。

embedding、LM head、router、视觉塔/连接器、MTP 以及非 A/B 学习状态也列入清单，不隐含宣称支持；它们的边界见第 5 节。真实 adapter 到位后，必须重新逐张量核对，实际文件中出现的学习张量不能因不在合成资产内而被忽略。

## 3. 当前模块支持矩阵

状态含义：“已验证”只对已测配置成立；“已有实现”表示静态代码路径存在；“需修改”有明确代码缺口；“待验证风险”需要最小复现，不能当作已确认运行故障。

| ID | 逻辑目标与运行位置 | 当前实现状态 | 已有 M3.1 验证 | 后续动作 |
| --- | --- | --- | --- | --- |
| A1 | dense `self_attn.q/k/v_proj` → 三段 `qkv_proj` | 已有通用/Ascend QKV wrapper | 已验证，限既有单卡范围 | 保留回归；采用五项模型映射后补 dense 兼容 |
| A2 | sparse Q/K/V → 五段 `qkv_proj` 的前三段 | 个人分支专用 wrapper | 已验证，限既有单卡范围 | 扩展到五段；验证原三目标 adapter 不回退 |
| A3 | `self_attn.o_proj` → RowParallelLinear | 已有 Ascend row wrapper | 已验证，限既有单卡范围 | 补 TP 输入切片、归约和多 adapter 路径 |
| I1 | `index_q_proj` → 五段 `qkv_proj` 第四段 | **需修改**：现有 LoRA 映射未并入融合层，wrapper 只更新前三段 | 未验证 | 补映射、第四段 buffer、加载、计算和 KV 式 TP 分片 |
| I2 | `index_k_proj` → 五段 `qkv_proj` 第五段 | **需修改**：同上 | 未验证 | 补第五段；每个 TP rank 加载相同 index_k 源权重 |
| M1 | 普通 `mlp.gate_proj/up_proj` → `gate_up_proj` | 已有 merged-column packed wrapper | 未验证 | 先真实加载/数值验证，不预设必须新写 MLP wrapper |
| M2 | 普通 `mlp.down_proj` → RowParallelLinear | 已有 Ascend row wrapper | 未验证 | 验证激活后输入、TP 切片、scale 和 reset |
| S1 | `block_sparse_moe.shared_experts.gate_proj/up_proj` | 与普通 MLP 复用；MoE wrapper 保留共享专家路径 | 未验证 | 核对别名引用均指向同一 wrapper，避免遗漏或重复增量 |
| S2 | `shared_experts.down_proj` | 已有 dense row 路径 | 未验证 | 验证与 routed 输出相加时仅应用一次 |
| E1 | `experts.{e}.w1/w2/w3`，每专家独立 2D A/B | 已有 `AscendFusedMoEWithLoRA`、加载/TP/EP 切片及 BF16 GMM 注入路径 | 未做 M3.1 专家 adapter 实机验证 | 对每个专家轴、w1/w3 顺序、路由权重和 combine 验证 |
| E2 | 仅部分专家或仅 w1/w2/w3 之一存在的 2D adapter | **静态限制明确**：现有 pack_moe 对每个专家要求 w1/w2/w3 均非空 | 未验证 | 先保存公开加载入口失败；缺失目标须按零增量语义支持，或明确报告不支持，不静默遗漏 |
| E3 | 磁盘已融合的 gate-up/down 专家格式 | 已有 `FusedMoE3DWithLoRA` 与 mixed-format 管理路径，格式及配置有条件 | 未验证 | 单列格式准入、专家轴、gate/up 排列和 EP 测试 |
| E4 | shared-outer：专家之间共享某些 A/B 因子 | 上游已有配置/分配；Ascend gather 路径的广播兼容待验证 | 未验证 | 单列风险，不能借“共享专家支持”推断该格式支持 |
| C1 | attention + indexer + MLP + shared + routed 同一个 adapter | 各路径需要组合核查 | 未验证 | 必需组合验收，不能只累加单模块通过结果 |

“共享专家”是每个 token 都执行的独立 MLP；“shared-outer/shared LoRA factors”是不同路由专家共享某些低秩因子，二者不是同一功能。

### 3.1 支持判定的代码依据

以下行号对应上述固定 commit；修改后应更新引用。

| 依据 | 文件与位置 | 能证明的事实 |
| --- | --- | --- |
| QKV 现状 | [utils.py:139](../../vllm_ascend/lora/utils.py#L139)、`:150`、`:164` | 专用 wrapper 仅接收三项映射、只更新前三段 |
| 实际融合布局 | [msa_m3.py:1257](../../vllm_ascend/models/minimax_m3/msa_m3.py#L1257)、`:1285`、`:1301` | 物理五段、KV 复制、各 rank 的输出宽度 |
| BF16 融合选择 | [msa_m3.py:1572](../../vllm_ascend/models/minimax_m3/msa_m3.py#L1572) | quant_config 为 None 时使用融合 QKV/indexer；纯 BF16 不必为量化拆分分支另做实现 |
| 模型映射 | [minimax_m3.py:1488](../../vllm_ascend/models/minimax_m3/minimax_m3.py#L1488) | 当前 qkv 三项；indexer 另映射到 indexer_proj；gate-up 和专家声明存在 |
| 普通/共享 MLP | [minimax_m3.py:680](../../vllm_ascend/models/minimax_m3/minimax_m3.py#L680)、`:785` | merged-column gate-up 与 row down，激活为 M3 clipped SwiGLU |
| dense NPU 路径 | [utils.py:48](../../vllm_ascend/lora/utils.py#L48)、`:105`、`:212` | packed buffer、set/reset 与 dense wrapper；部分包装器限定 max_loras=1 |
| 专家名字 | [minimax_m3.py:1295](../../vllm_ascend/models/minimax_m3/minimax_m3.py#L1295) | w1=gate，w2=down，w3=up；不按普通 MLP 后缀猜测 |
| 专家包装/限制 | [fused_moe.py:166](../../vllm_ascend/lora/fused_moe.py#L166)、`:304`、`:359` | 已复用上游分配/切片，发布 NPU LoRA context；拒绝 dynamic EPLB/FusedMC2 |
| 实际专家增量 | [routed_experts.py:244](../../vllm_ascend/ops/fused_moe/routed_experts.py#L244)、`:307` | gate-up GMM 后、激活前加 w13；down GMM 后加 w2 |
| 专家算术边界 | [punica_npu.py:477](../../vllm_ascend/lora/punica_npu.py#L477)、`:530`、`:542` | 按 adapter/expert gather；shrink buffer 为 FP32 |
| 2D 部分专家限制 | [vLLM lora_weights.py:196](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/lora/lora_weights.py#L196)；model_manager.py:870 | 缺失项保留为 None，pack_moe 随后断言完整三投影 |
| 专家格式选择 | [vLLM model_manager.py:145](https://github.com/vllm-project/vllm/blob/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15/vllm/lora/model_manager.py#L145)、`:559`、`:930` | wrapper 选择受模型布局、mixed/shared 配置控制；不是只读模型 mapping 的三项长度 |
| 原测试范围 | [prepare.py:20](../../tests/e2e/models/minimax_m31_lora/prepare.py#L20)、[driver.py:117](../../tests/e2e/models/minimax_m31_lora/m31_lora_driver.py#L117) | 生成资产及运行 filter 均只包含 Q/K/V/O |

### 3.2 磁盘名称与格式必须单独核对

上表使用 vLLM 可见的逻辑名字，不能直接当作所有训练框架的 PEFT 名字。当前归档 HF 实现包含嵌套 `self_attn.indexer.q_proj/k_proj`、融合 `mlp.gate_up_proj`，以及专家 3D 参数；这些与 vLLM 的平铺 index_q/index_k、拆分 gate/up、每专家 w1/w2/w3 名称存在差别。

每个格式至少记录：完整原始模块名、磁盘 A/B key、rank/alpha、A/B shape、模型 mapper、运行模块、物理 slice/expert、各 rank 的源范围、最终 wrapper/buffer。原始名字保留在证据中。后缀 `q_proj` 既能指主注意力也能指嵌套 indexer，禁止按短名字混在一起。

| 格式 | 处理要求 |
| --- | --- |
| 平铺 `index_q_proj/index_k_proj` | 对应五段中的第四、第五段 |
| 嵌套 `indexer.q_proj/k_proj` | 查实际 adapter，必要时增加模型专用精确重命名；不能只改变测试文件名绕开真实格式 |
| 拆分 gate_proj/up_proj | 两套独立 A/B，可单独活动；未活动片段保持零增量 |
| 融合 gate_up_proj 的一个低秩对 | A 可共享、B 按输出行拆分；不得无依据复制 rank 或重复 scale |
| 逐专家 2D w1/w2/w3 | 明确 expert_id 和三投影；完整、单投影、专家子集分别验证 |
| 已融合/堆叠专家参数 | 按配套 manager 的真实 PEFT 契约处理；不能把普通 nn.Parameter 的任意 A/B 格式当作已支持 |
| shared-outer | 记录专家轴为 1 的共享因子；验证广播及 adapter/expert 联合索引 |

## 4. 参考上游的实施方案

推荐保持配套 c410 commit，移植必要的 M3 通用布局处理到 Ascend 专用包装器。另一种做法是在实验 vLLM 分支回移 PR 后让 Ascend 继承，但会产生非主仓配套依赖；本方案不切换该依赖。未来上游合入且配套升级后，再删除可被上游替代的部分。

### 4.1 必改：indexer 五段与 dense 兼容

参考 [PR head 的 custom_layer.py](https://github.com/aoshen02/vllm/blob/4917985161655ab0ca549f8b6e07ead88338d078/vllm/lora/layers/custom_layer.py#L93)。

1. 在 LoRA 启用的模型实例上设置五项 qkv 映射；text/VL 顶层及 language_model 的实例配置保持一致。保留量化所用的类级声明，不将 BF16 修复扩大到量化分支。
2. 五段 wrapper 继承已有 Ascend packed 执行，使用真实 base_layer.output_sizes/tp_size；Q 按 q_shard_id，K/V/index_q 按 kv_shard_id，index_k 的源 shard 恒为 0。
3. n_slices、output_slices、output_ids、output_sizes、A/B buffer 及 packed buffer 必须同时扩展；不能仅改 mapping，也不能继续只更新输出前三段。
4. dense 层只有 Q/K/V，五项逻辑映射中的两个 indexer 项必须为空。参考上游 dense wrapper 验证后仅传前三项，不能静默丢弃错误存在的 indexer 权重。
5. 保留旧 Q/K/V/O adapter、indexer-only、任意有效部分目标的加载/reset；新 wrapper 优先于可误匹配的通用类，原有非 M3 包装器行为不变。
6. 当前上游参考拒绝 fully_sharded_loras。M3 入口应清楚报告该限制，不以跳过包装继续推理；普通 TP 不等于 fully-sharded LoRA。

### 4.2 MLP/共享专家：优先验证和复用

不预设新增生产包装器。验证普通 MLP 与共享专家的 gate/up slice、down TP 输入切片，以及 LoRA 加在 clipped SwiGLU 之前/之后的位置。检查 MoERunner 对共享专家的别名、包装后的调用路径与加载名一致，确保只执行一次。发现真实格式别名不兼容后，再按明确模块修正 mapper。

### 4.3 路由专家：复用现有 NPU MoE 链路

复用上游分配、set/reset、TP/EP 切片及 Ascend context/GMM 注入。两条增量位置必须保持：

```text
per-expert input → base w13 + LoRA(w1,w3) → clipped SwiGLU
                 → base w2  + LoRA(w2) → routing weight/scale → combine
shared_experts(input) --------------------------------------→ add
```

完整 2D 专家格式先验证；对部分专家/投影先用真实公开加载入口保存失败，再做最小修复。未声明整个目标表示零增量；只有 A 或只有 B 属于损坏配对，必须拒绝，二者不能混淆。不能通过在测试资产中悄悄补齐全部非零权重规避部分目标问题。

部分目标限制位于上游 manager/packer，而非 Ascend GMM 内核。先核对可复用公共扩展点；如配套没有合适扩展点，应明确记录“保持纯配套主仓时该格式受限”和所需的上游补丁，不能声称只改 Ascend set_lora 就能解决。

shared-outer 另做最小复现：上游允许某些 buffer 专家轴为 1；当前 NPU 算法按 adapter×expert 索引展平的 A/B。验证两者兼容前不宣布支持；若失败，修正共享因子的索引或明确拒绝配置。此风险不等同“共享专家 MLP 不支持”。

### 4.4 需要复现后决定的兼容项

- 基座重载：参考 PR 处理 `.base_layer.` 名字变化。当前配套已有 `BaseLayerWithLoRA.load_weights`，先复用生命周期接口，核对 M3 手写 loader 的实际参数名；不能原样复制旧 PR 的全参数扫描。
- VL 入口：参考 PR 显式给出 language_model 的 get_mm_mapping。Ascend 当前有 `_mark_language_model/_mark_tower_model`，没有自定义 get_mm_mapping；文本-only 成功不能证明带图输入的 adapter 元数据正确。
- 多 adapter：部分 Ascend packed wrapper 的 can_replace 限定 max_loras=1；max_loras=2 可能选择另一通用路径。必须确认实际选择和调用的 NPU kernel，不能只删除这个条件。

以上是待验证项，不作为本轮新确认的运行缺陷。

## 5. 其他候选目标与明确边界

| 目标/配置 | 当前代码判断 | 本方案的验收边界 |
| --- | --- | --- |
| embedding | vLLM 有专用 embedding LoRA、NPU 有执行接口 | M3 未验证；真实 adapter 涉及时增加词表边界/共享基座检查，不并入现有通过声明 |
| LM head | 有 logits/head LoRA 路径 | M3 未验证；增加 sampler/logits 位置、padding 裁剪和 tied-weight 检查 |
| router GateLinear | 配套 ReplicatedLinearWithLoRA 可匹配 GateLinear 子类 | NPU、M3、FP32 输出及内部路由旁路均未验证；本次常规专家 LoRA 保持 router 冻结，若实际 adapter 含 gate 则专项纳入 |
| router bias、RMSNorm 等非 A/B 参数 | 不属于普通低秩线性适配 | 需要额外学习状态契约；不能过滤后声称整个 adapter 正确 |
| 视觉 QKV/O、fc1/fc2 | ViT 有通用线性类，缺少本轮多模态 LoRA 证据 | 单列扩展目标；语言侧 LoRA 需有“冻结视觉塔＋图文输入”兼容用例 |
| projector / patch merger 的 linear_1/linear_2 | 有普通 column/row 基础层 | 未验证；需要 tower/connector 映射和图像 token 路由，不能由语言层结果推出支持 |
| MTP/draft | 基座 loader 可跳过相关权重，不代表 adapter 支持 | 主闭环关闭；专项检查请求启用时拒绝/支持的实际行为，不能静默忽略其学习权重 |
| DoRA、额外 modules_to_save、特殊因子共享变体 | BF16 只是 dtype，不自动保证这些变体 | 标准 LoRA 为主合同；出现这些字段时显式分类、验证或拒绝 |
| fully-sharded、dynamic EPLB、FusedMC2 | 当前有模型/后端限制 | 负向准入用例；不使用这些配置作为默认通过路径 |
| FP8/INT8/INT4/NVFP4、量化 KV cache | 不在 BF16 本轮范围 | 不新增这些数值矩阵；BF16 模式应核对实际 cache dtype |

BF16 指基座线性/专家权重、LoRA 执行因子和主激活为 BF16；router 权重/修正 bias、归一化累积及部分 LoRA workspace 按模型和算子契约保留 FP32。不得把“BF16 场景”解释成把这些张量强制转换为 BF16。

## 6. 测试资产与不变量

### 6.1 两级模型

1. 开发资产使用真实 checkpoint 减层，至少覆盖 dense attention、sparse attention、普通 MLP、共享专家及 routed MoE。保留 hidden/head/专家宽度、专家数/top-k、router、词表及原始张量值；每类实际存在性必须由配置和 runtime inventory 证明。
2. 现有前 4 层资产先审计；若缺少某类，不用“这四层通过”替代该类覆盖。若 checkpoint 本身没有某可选结构，标记 checkpoint-missing；可用小配置补 UT，但不能宣称真实 M3 该结构已通过。
3. 最终用完整原始层数做真实权重回归。完整模型验收使用可容纳模型的 TP；不要求全模型必须单卡。减层仅证明相应结构和实现，不能证明全层正确性。

### 6.2 Adapter 分组

每组生成 zero、非零 A、非零 B。A/B 为两个 adapter 名称，每个 adapter 内仍包含低秩因子 A 与 B；两者需要按层/slice/expert 的不同 seed 和指纹区分。

| Profile | 必需目标与主要目的 |
| --- | --- |
| attention-q/k/v/o、attention-all | 独立投影及现有范围回归；dense/sparse 均覆盖 |
| index-q、index-k、index-all | 分别定位第四/第五段；不得依赖 QKV 有增量掩盖 indexer 无增量 |
| attention-index-all | 五段＋O 同时活动，含 dense 层无 indexer |
| mlp-gate、mlp-up、mlp-down、mlp-all | 普通 MLP 部分目标与组合 |
| shared-gate、shared-up、shared-down、shared-all | 共享专家独立执行和与 routed 合并 |
| expert-w1、expert-w2、expert-w3、expert-all | 每个专家投影的增量；部分目标限制必须显式暴露 |
| expert-subset | 专家 0、最后一个、非连续专家子集；未活动专家保持零增量 |
| expert-3d、expert-mixed、expert-shared-outer | 单列格式扩展，不与标准 2D 的通过数混算 |
| core-all | attention＋indexer＋MLP＋shared＋routed 的完整组合 |

rank=8 为主，rank=16 覆盖不同 adapter rank/slot padding；对实际支持的较大 rank（如 128）选择一次不同 NPU 分支的代表回归。alpha/rank 至少两个非单位缩放，保持标准低秩，不使用 merged 缓存替代运行分支。

生成阶段独立检查非零信号，避免落入原容差以内。信号无效记 INVALID_PROBE，新建资产和 run；不依据失败输出调整容差、删除目标或把训练初始化 B=0 作为唯一测试。

## 7. 分层验证矩阵

所有行起始状态为 PLANNED。原 68 项仅作为旧合同回归，不自动填充下表的新 case。

| 层级/ID | 必测内容 | 判定与证据 |
| --- | --- | --- |
| T0 inventory | 两个入口；逐层模块、真实类、target filter、磁盘格式、mapper、wrapper、A/B buffer | 每个学习张量有归属；全量 key/shape/hash；无未解释 missing/skip |
| T1 selector | 五段/三段、三目标旧格式/五目标新格式、indexer-only、MLP、MoE，max_loras=1/2 | 使用真实构造器和 from_layer；错误类型不替换，正确类型必须替换 |
| T2 loader | shape、rank、scale、转置、slice、专家轴、TP 源范围、packed buffer | 与磁盘因子独立计算一致；未活动项为零；scale 恰好一次 |
| T3 kernel | NPU 实际 dense/packed/MoE 增量，prefill、至少两步 decode | 独立参考对实际 X 和磁盘 A/B；同时验证增量、最终层输出、未活动片段 |
| T4 model | base、zero、A、B、所有 profile、core-all | 固定 prompt/token 历史，独立 CPU 参考；真实模型 logits/logprob/token 与首个分歧 |
| T5 merged | 每组 adapter 独立合并基座 vs base＋LoRA | greedy token ID 必须完全一致；失败保留数值证据，不降级为文本相似 |
| T6 lifecycle | A→base→B→A；load/remove/load；从全目标切到部分目标；slot 回收 | token、buffer 清零、基座内容和共享关系；无残留增量 |
| T7 service | base/A/B 同时处理 vs 各自独立处理 | 同配置、同 prompt 的 token ID 完全一致；保留同批和 adapter 索引证据 |
| T8 parallel | TP、DP、EP、TP＋DP＋EP | 每个 rank 的 buffer/路由元数据、集合通信及最终输出；见第 10 节 |
| T9 full/device | 完整层数、A3 和 A5 分别执行、普通无 hook 控制 | 分硬件/配置签收；不将模拟、CPU 或另一硬件结果算作本机通过 |

### 7.1 Indexer 专项

- 对固定同一层输入，分别只启用 index_q、只启用 index_k、同时启用；Q/K/V 等未活动 slice 精确不变。
- TP 小于 KV 头数时覆盖真实分片，大于 KV 头数时覆盖复制；index_k 所有 rank 源 slice 相同。
- 验证 norm/RoPE 后的 index query/key、cache 写入以及最终 top-k block 集合。score 接近边界时保留选择 margin 和首个不同 block；不能只看最终 token。
- prompt 长度覆盖 block_size−1、block_size、block_size+1；另外选择超过实际 top-k/local-block 覆盖容量的长上下文，使候选块确实多于可选块。
- 使用独立信号诊断保证至少一个非零 indexer probe 改变 score，并在预先冻结的长上下文探针中改变选择集合。只测短 prompt 时，“输出相同”不能证明 indexer LoRA 生效。

### 7.2 普通 MLP 与共享专家专项

- gate/up 独立 LoRA，使用不同因子，能检测 slice 对调；验证 clipped SwiGLU 前注入。
- down 使用实际激活输出作为输入，验证 TP 分片、归约及非单位 scale。
- profile 切换时验证未活动的另一片段清零；shared-only 不修改 routed 专家权重。
- shared+routed 同时活动时，分别记录两分支和最终和；同一个共享模块的别名不能重复施加 LoRA。

### 7.3 路由专家专项

- T2 对所有声明专家逐个核对 buffer；T3 使用可控路由的层级输入覆盖每个专家，而非寄希望于少量整模 prompt 命中全部专家。
- T4/T5 使用真实 router，记录专家选择和归一化 routing weight；模块组合变化导致上游 hidden state 不同，不要求它与 base 选择相同专家。
- 验证 w1/w3 顺序、M3 非交错布局、clipped SwiGLU、w2 输入和 routed_scaling_factor；路由权重只乘一次。
- EP 覆盖远端专家、同 token 多专家、空本地专家/零派发 token、非均匀 token 分布、base token 的 adapter index=−1。
- 对部分专家、部分投影分别先测真实文件缺失与语义零增量；不是只用含全套权重而把某些 B 设零来替代格式准入。
- 3D/shared-outer/mixed-format 分开记录，每种都需实际 loader 到设备的证据。

## 8. 数值参考与用户验收线

层级主参考继续使用已确认的独立 BF16 执行参考，并保留理想 FP64 差异报告。沿用已冻结的 `atol=rtol=1/64`，不因新增模块失败放宽。结构、key、slice、expert/adapter index、未活动片段和确定性行为要求精确匹配。

dense 单 slot 的 matmul 路径与 MoE/multi-adapter kernel 的中间存储边界分别建参考：现有 dense shrink 可存 BF16；MoE 的 shrink buffer 为 FP32。参考依据实际 primitive 契约和独立小算例建立，不从待测输出拟合。CPU 自然执行顺序不自动等价 NPU 融合运算。

**用户验收线一：模型正确性。** 从原始磁盘因子独立复现固定的 dtype 转换与缩放边界，得到 A_eff/B_eff；若 scale 已折入 B_eff，不再乘第二次。构造 `W_merged = BF16(FP32(W) + FP32(B_eff) @ FP32(A_eff))`，共享因子/融合输出/专家轴按真实语义展开，保持 router 等非目标状态和必要 FP32 参数不变。不能一边使用磁盘 FP32 因子、一边使用转换后的 BF16 因子作为同一 adapter。将 merged 模型作为无 LoRA 模型推理，与同一并行配置下的 base＋LoRA 比较 greedy token ID，要求完全一致。

BF16 合并权重和在线两次 GEMM 的舍入顺序不同，token 严格一致不是数学恒等式。若失败，仍记该验收线 FAIL，保存首个分歧 token、logit margin、logprob 和层级结果，区分映射/实现错误与合并舍入差异；不能把层级容差通过改写为 token 验收通过，也不能为追求相同 token 降低 adapter 幅度。

**用户验收线二：服务正确性。** base/A/B 混批与各自单独请求的 token ID 完全一致。单独请求基线必须来自相同服务配置、同一 max_loras、多 slot kernel、同样的图/cache/并行设置；不能拿 max_loras=1 的另一执行路径当唯一基线。

温度固定 0，seed 固定，prompt 使用相同 token ID，保持 tokenizer、模板、生成长度和 stop 规则一致；至少含中文、英文、代码、重复 token、不同长度和长上下文。部署准确性还需代表性真实 prompt；合成 adapter 只能证明实现，不证明微调质量。

## 9. 混批服务方案

参考 skill 的 P0–P9 明确是单请求合同，不承担本节；保留原合同，通过独立 pytest/服务用例验证混批，不修改旧状态机假装它有此覆盖。

1. 以 max_loras=2、max_cpu_loras≥2、max_num_seqs≥3 同时驻留 A/B；base 不占 adapter slot。关闭 prefix cache/chunked prefill，使用已核查的 batch-invariant 设置作为严格一致性基线。
2. 先采集 base、A、B 的单请求基线，再用屏障同时提交 `[base,A,B]`、`[A,base,B,A]`、交错长短 prompt，覆盖同时 prefill、交错 prefill/decode、多步 decode。
3. HTTP 使用配套 `/v1/completions` 的 `return_token_ids=true`，读取原始 token_ids；不得把输出文本重新 tokenize 当作原始 ID。
4. 同时提交不保证同批。保存实际 scheduler batch、每个请求 ID 和 token 的 adapter slot/−1 映射；必须至少观察到同一执行批次混有 base/A/B，否则该 case 为 UNAVAILABLE，不能 PASS。
5. 用确定性调度边界/事件构造顺序，禁止靠固定 sleep 碰运气。采集采用最小诊断能力，并补普通无 hook 推理对照，避免观测改变行为。
6. 不同 profile 的 A/B、不同 rank 的 A/B 和全目标→部分目标组合均做至少一个代表性混批回归。
7. 另做 slot 回收/重载：用第三个 C 触发两 slot 的重新分配，再恢复 A/B；记录 slot 布局变化后的元数据更新和原 adapter 输出恢复。
8. 显式启用/关闭图、cache、chunked prefill 的兼容用例独立列示；这些附加配置失败不能反向删去 eager 主场景。

## 10. BF16 并行与硬件矩阵

PCP/PP 固定 1。EP 是专家分布开关，不是与 TP、DP 再相乘的独立卡数。本配套在启用 EP 后可将 TP×DP 设备组成专家组，MoE 内部 TP 变为 1；实际组大小需由运行日志确认。

| 配置 | 计划设备数 | 主要目的 |
| --- | --- | --- |
| TP1 / DP1 / EP off | 1 | 全部核心 profile 的层级、整模、生命周期及混批 |
| TP2 / DP1 / EP off | 2 | QKV/indexer/MLP/expert 权重张量分片与输出归约 |
| TP4 / DP1 / EP off | 4 | 实际 KV=4 时的分片边界；配置不同则按真实头数调整代表点 |
| TP8 / DP1 / EP off | 8 | 实际 KV=4 时的 KV/index_q 复制分支 |
| TP1 / DP2 / EP off | 2 | DP 请求分发、各 rank 的一致加载与缓存隔离；核对 MoE 实际分组 |
| TP2 / DP1 / EP on | 2 | 专家 ownership、跨 rank dispatch/combine |
| TP1 / DP2 / EP on | 2 | 跨 DP 的专家派发与 adapter index 传递 |
| TP2 / DP2 / EP on | 4 | 用户要求的 TP＋DP＋EP 联合模型和服务验收 |

TP1 对照是定位基准；merged/LoRA 的严格验收优先在同一 TP/DP/EP 配置内进行。跨 TP 单列 token、logprob 及首差异报告，不能将不同归约顺序直接归因为 LoRA 错误。

A3、A5 分别执行基准、indexer 长上下文、普通/共享/路由专家、core-all、merged、混批和联合并行代表项。一个硬件的通过不能签收另一个。没有 A5 设备时保持 A5 未验证，不替换为模拟结果。

MoE 开始时关闭 dynamic EPLB/FusedMC2，使用线性专家 placement。配套上游 `_ep_check` 对 backend/SP 有限制；Ascend AllGather/AlltoAll 是否满足实际配置需在 T0 准入核查，保留实选通信路径，不只记录命令行 flag。现有上游断言拒绝的配置先报告，不直接删断言。

开发阶段不用所有 profile 与全部并行配置笛卡尔积：单卡覆盖每个核心 profile；每种分片语义做真实层对照；TP/DP/EP 的整模与混批至少覆盖 attention-index、MLP/shared、expert、core-all。完整模型使用容量可行的配置完成这四组代表与用户两条验收线。

## 11. 实施任务与文件归属

每项先保存当前提交的失败/限制，再修改并复测。测试名和新文件名为计划，尚不存在的项不能作为已执行命令。

| 任务 | 生产/测试位置 | 交付条件 |
| --- | --- | --- |
| D0 盘点/冻结合同 | 现有 `tests/e2e/models/minimax_m31_lora/prepare.py` | 动态读取配置、输出完整 module/format inventory；没有未解释学习张量 |
| D1 五段包装 | `vllm_ascend/lora/utils.py`；扩展 `tests/ut/lora/test_minimax_m3.py` | 五段和 dense 三段均能选择、加载、计算；旧 QKVO 回归 |
| D2 实例映射/入口 | `vllm_ascend/models/minimax_m3/minimax_m3.py`、`minimax_m3_vl.py` | text/VL 实例映射一致；必要的精确 mapper；fully-sharded 清楚拒绝 |
| D3 MLP/shared 验证 | 扩展 prepare/driver/observation；模型专用验证放 M3 目录 | 实际 gate/up/down 及共享专家证据；仅修复复现出的生产缺口 |
| D4 MoE 验证 | 现有 `vllm_ascend/lora/fused_moe.py`、`punica_npu.py`、`ops/fused_moe/routed_experts.py` 为候选修改位置 | buffer、路由、FP32 shrink 参考及 GMM 两处注入证据；不预设改内核 |
| D5 专家格式/部分目标 | 先检查上游 `lora_weights.py`、`model_manager.py` 公共实现 | 区分基线限制与本次引入；按可用扩展点最小处理或明确补丁依赖 |
| D6 数值/merged | 新建 `tests/e2e/models/minimax_m31_lora/test_bf16_correctness.py`；复用资产和独立参考 | 所有核心 profile 与 core-all 的 T2–T6，含真实权重公开 LLM 入口 |
| D7 混批服务 | 新建同目录 `test_bf16_service.py`，参考现有 two_card/lora MoE 服务用例 | HTTP 原始 token、真实同批证据、base 隔离及 slot 生命周期 |
| D8 并行/设备 | 参数化 D6/D7 并接入适当 CI 选择 | A3/A5、TP/DP/EP 分项报告；资源不足明确记阻塞 |
| D9 审计/交付 | 新验证报告、manifest、CI/格式检查、中文使用说明 | 只声明有当前有效证据的目标/格式/配置；不覆盖旧报告 |

通用 packing/通信行为 UT 放对应通用生产模块测试；M3 映射 UT 放 M3 测试。新增用例使用真实构造器，mock 仅限外部服务，不用 SimpleNamespace 伪造核心层证明支持。测试辅助方法不额外套一层低收益 UT。

执行命令遵循仓库环境要求，以下第一条是现有 UT 的命令；后两条待 D6/D7 文件落地后执行：

```bash
cd /vllm-workspace/vllm-ascend
TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv/bin/python -m pytest -q tests/ut/lora
.venv/bin/python -m pytest -v -s tests/e2e/models/minimax_m31_lora/test_bf16_correctness.py
.venv/bin/python -m pytest -v -s tests/e2e/models/minimax_m31_lora/test_bf16_service.py
bash format.sh ci
```

`.venv` 是后续实施时通过 uv 创建并绑定容器现有配套包的环境要求；不表示当前容器已有这个路径。使用实际 NPU/模型资源 fixture，按卡数分 CI/串行执行；不要临时升级 Transformers 或替换配套 vLLM。

## 12. 最终审查和签收

| 视角 | 计划验证 |
| --- | --- |
| 安全 | adapter 只读预期目录/格式；不可信 key、损坏文件明确拒绝；公开报告无凭证、权重、私有激活或完整日志泄露 |
| 性能/资源 | rank、专家数、长上下文和 slot 有预算；热路径无全参数扫描；大采集只用于诊断；本方案不承诺性能收益 |
| 边界 | 空专家组、未活动 slice、全零 B、缺 A/B、错 shape、非法 rank、不同 rank slot、KV 复制和词表边界 |
| 时序 | slot 回收、remove/reload、元数据刷新、失败后恢复、跨 EP 派发一致；等待明确事件/状态，不固定 sleep |

每项问题报告具体文件/行号、触发、影响和反驳检查；当前仅为静态风险的项目需标注，不能伪装成已复现故障。保留修复前证据与新构建身份，改代码/合同后使旧结果失效或新建 run，禁止混算。

最终分别交付：模块/磁盘格式支持表、单请求数值结果、merged 严格 token 结果、混批严格 token 结果、并行矩阵、A3/A5 矩阵。PASS、FAIL、INVALID_PROBE、UNAVAILABLE 分开统计；negative case 的“正确拒绝”只能证明边界处理，不能计作对应功能已支持。

只有核心模块的声明格式通过真实加载、设备增量、整模、生命周期及适用的两条用户验收线，才能声明该配置完成。没有实际测试的新目标仍为未验证；不能以旧 68 项、CPU UT 或上游作者报告替代。
