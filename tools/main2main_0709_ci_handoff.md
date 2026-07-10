# main2main 0709 CI 临时交接记录

> **临时文件：本文件仅用于 PR #11709 定位与交接，正式合入前必须删除，不得随功能提交进入主分支。**

## 1. 分析基线与已确认范围

- 下游 PR：`vllm-project/vllm-ascend#11709`，当前分析 head 为 `838f1aa8b81a65668a9b4566e6c8a70613900655`。
- 当前目标 vLLM commit：`ab7961a14a59be9a0170f1654315d5c2be44c015`。
- 已知可以通过精度用例的 vLLM 边界：`ba22152096b2484faa3579624a253d54804d876d`。
- `ba221520` 是 `ab7961a` 的祖先；待适配区间为 `(ba221520, ab7961a]`，共 155 个上游提交。
- 当前已证明会触发问题的上游修改均处于该区间：
  - `vllm#47668` / `07f9baf7564b42ba7218ce9167bfcc4128028473`：多个运行期事件从 `torch.Event` 恢复为 `torch.cuda.Event`，覆盖 MRV2、spec decode、LoRA、multi-stream、offload 等路径。
  - `vllm#47081` / `d3e69fd6714e9d1bb6e8e4f03157090dc32e7960`：MRV1/MRV2 异步输出事件改为 `torch.cuda.Event(blocking=True)`。
  - `vllm#47408` / `5d23ca47ab597d9...`：MoE `routed_scaling_factor` 语义发生变化，暴露下游 native fallback 重复缩放问题。
  - `vllm#41359` / `dd944845777b303000a2e153707382cf03383e26`，以及 `vllm#47872` / `51e5372f3d8...`：Transformers/tokenizer/Hunyuan processor 契约变化。
  - `vllm#47970` / `e7b3853...`：DeepSeek-V2 router dtype 行为变化，只作为 Event/MoE 修复后的残差候选。
- 准确归属：不是所有错误都表示上游实现本身错误；主要是该区间内上游接口/语义变化进入了下游原有 patch 的不完整语义路径。

## 2. 当前依赖策略

- **本轮保持 Transformers `5.5.4`，暂不升级。**
- 目的：先单独收敛精度、异步 Event、MoE scaling 等问题，避免 Transformers 升级同时改变 tokenizer、guided decoding 和多模态 processor 行为，造成归因混杂。
- 当前安装过程存在“vLLM 安装 5.13.0，vllm-ascend 随后降回 5.5.4”的状态，这是已知版本契约差异，但本轮不得顺手升级。
- 在 Event 修复后若只剩 guided decoding/Hunyuan 相关失败：
  - guided decoding 的 Transformers 兼容残差单独记录，留到依赖升级阶段处理；
  - Hunyuan 如必须先清理 CI，只允许使用独立兼容提交修复 vLLM lazy registry，不得与精度修复混合。

## 3. 已提取失败矩阵

基于 Actions run `29088445112` 已完成失败 job 的日志，当前确认 9 个失败 job、24 个失败测试项：

| Job | 主要失败 | 观察到的信号 | 当前优先归因 |
| --- | --- | --- | --- |
| `86348980841` | MRV2 DSpark Qwen3-8B | acceptance 明显低于 golden | MRV2 独立定位；不能用 MRV1 placeholder 直接解释 |
| `86348980858` | CPU weight prefetch eager/ACL graph | `KeyError: 279`；负数垃圾 token；tokenizer `OverflowError` | async copy/Event；测试 baseline 配置另有缺陷 |
| `86348980859` | xgrammar/guidance/outlines | 非法 token 0/13/15、`!`、空串、FSM reject、EngineDead | 首先检查异步 sampled-id 回传；Transformers 残差后置 |
| `86348980861` | Hunyuan-VL | 删除模块路径导致 `ModuleNotFoundError` | 独立 processor registry 兼容问题，非精度主因 |
| `86348980863` | 4-card CP，7 项 | 重复/漏 token，例如 `man man`、`Paris Paris` | async output/Event；Event 后再看 MoE/CP 残差 |
| `86348980869` | iLLaMA LoRA TP2、DSv2 prefix cache、shared-expert DP | `count!(*)`；末 token `0/17`；sampled id 不在自身 logprobs 中 | Event 最强 canary |
| `86348980871` | Qwen Eagle3、multistream shared expert | acceptance 异常；graph/multistream 最后 token 改变 | Event；acceptance 需固定 seed 多次验证 |
| `86348980885` | Qwen3-VL SP TP2 | SP on/off 首 token 明显异常 | Event；修复后再判断 SP 语义 |
| `86348980905` | Llama3.2 LoRA TP2、Qwen3 VWN Eagle3 | 重复/漏词；acceptance 偏低 | Event；LoRA 内部 Event 路径需覆盖 |

共同特征：已完成的生成/精度失败均运行在 async-enabled 模式，异常集中于 host 可见的 sampled token、logprobs 及其 scheduler 消费点。但不能据此表述为“开启 async 的大部分用例都会失败”：同一批日志中有大量 async-enabled 通过项，是否失败取决于具体 Event 创建位置、copy/读取时序和断言强度。

## 4. MRV1/MRV2 Event 边界

### 4.1 可通过边界 `ba221520`

- MRV1 `vllm/v1/worker/gpu_model_runner.py::AsyncGPUModelRunnerOutput` 使用 `torch.Event()`。
- MRV2 `vllm/v1/worker/gpu/async_utils.py` 使用 `torch.Event()`。
- 下游初始化期间设置的 `torch.Event = torch.npu.Event` 能提供真实 NPU 事件语义，因此 `ba221520` 即使开启 async scheduling 也可以正常同步。

### 4.2 MRV1 失败边界 `ab7961a`

- MRV1 的异步 output/pooling/prepare-input 等路径开始运行期创建 `torch.cuda.Event(blocking=True)`。
- #47668 涉及的 spec decode、LoRA、offload、multi-stream 等路径也开始创建 `torch.cuda.Event`，需要按其实际 worker 生命周期分别审计。
- 下游 `vllm_ascend/worker/model_runner_v1.py::_torch_cuda_wrapper()` 只在 `super().__init__` 期间临时把 `torch.cuda.Event` 映射为 `torch.npu.Event`；退出 wrapper 后又恢复为 `_EventPlaceholder`。
- `_EventPlaceholder.record/wait/synchronize` 均为 no-op，`query()` 永远返回 true。
- `AsyncGPUModelRunnerOutput` 等对象在初始化结束后的运行期创建，因此实际得到 no-op Event；nonblocking NPU 到 pinned CPU copy 未完成时，host 已经读取 sampled ids/logprobs。
- 这能统一解释：token 0/1、负数垃圾 token、随机 `!`、重复/漏 token、sampled id 与 logprobs 不一致、grammar 拒绝刚采出的 token。

### 4.3 MRV2 边界

- MRV2 使用独立的 `vllm_ascend/worker/v2/utils.py::torch_cuda_wrapper()`。
- 该 wrapper 将 `torch.cuda.Event` 映射为 `torch.npu.Event` 后，`finally` 不会恢复 placeholder，因此 MRV1 的 no-op Event 因果链不能直接套用到 MRV2。
- DSpark acceptance 失败必须在 MRV1 Event 修复后单独复测；若仍失败，应检查 PR #11709 的 rejection-sampler API 适配和 `(ba221520, ab7961a]` 中的 DSpark/spec 专属变化，必要时只对该用例做 bisect。

## 5. 修复计划

### P0：最小 A/B 留证据

固定 PR head、vLLM `ab7961a`、Transformers 5.5.4、模型、seed、max tokens，只改变 `async_scheduling`：

1. shared-expert DP：检查 sampled id 是否存在于同一步 logprobs。
2. DSv2 prefix cache：检查 prefix on/off 最后 token 的 `0/17` 差异。
3. guided xgrammar JSON 可作为第三个观察项，但不能单独用于 Event 归因。

判定：

| 当前 async on | async off | Event 修复后 async on | 解释 |
| --- | --- | --- | --- |
| 失败 | 通过 | 通过 | Event 根因闭环 |
| 失败 | 失败 | 通过 | scheduler 外部的 LoRA/offload/multistream Event 也被修复 |
| 失败 | 通过 | 失败 | Event adapter 不完整或仍有独立 async 改动 |
| 失败 | 失败 | 失败 | 转入独立残差分析 |

关闭 async 只作为定位手段，不得作为正式修复。

### P1：Event-only 最小修复

- 使用独立提交，仅修 Event 语义，不同时修改 Transformers、MoE scaling 或 golden。
- MRV1 worker 整个生命周期内，`torch.cuda.Event` 必须返回真实 `torch.npu.Event`。
- 当前 torch-npu 的 `Event` 原生接受 `blocking=True`；本轮直接保持真实映射，不新增参数吞噬或全局同步等防护逻辑。
- 禁止在初始化结束后恢复成 no-op `_EventPlaceholder`。
- 审计 #47668/#47081 涉及的 MRV1、pooling、prepare-input、draft/spec、LoRA、offload、multi-stream、KV connector 运行期 Event 创建点；MRV2 作为独立路径记录结果。
- 添加一个强制延迟的 nonblocking NPU 到 pinned CPU copy 测试：sync 前 copy 未完成，sync 后数据稳定且正确。

### P2：Event 修复后的定向复测

- 先重跑 P0 canary，再跑第 6 节代表矩阵。
- 不允许因一次 acceptance 通过就更新结论；不得更新 golden 或放宽阈值。
- Event 修复应首先消除垃圾 token、sampled-id/logprobs 不自洽、grammar 非法 token 等不变量错误。
- DSpark 不作为 MRV1 Event 修复是否成立的门槛；其结果单独归档。

### P3：残差分治

- 若只剩 DeepSeek/MoE 类失败：独立修复 `experts_selector.py` native fallback 对 `routed_scaling_factor` 的双乘，确保 fused/native/custom 路径 exactly once。
- 为 scaling 添加不依赖生成结果的纯权重单测，覆盖 factor 不等于 1、grouped/non-grouped/custom/fallback。
- #47970 router dtype 只在 Event 与 scaling 修复后仍有 DeepSeek 残差时检查。
- 若只剩 guided/Hunyuan：保持 Transformers 5.5.4，记录为依赖/processor 兼容阶段问题，不与本轮精度提交混合。

## 6. 验证矩阵

| 层级 | 用例/路径 | 目的 | 通过标准 |
| --- | --- | --- | --- |
| Event 单元 | 延迟 nonblocking NPU→pinned CPU copy | 验证真实 `record/synchronize/query` | sync 后数据必然正确；不得出现 placeholder |
| MRV1 canary | shared-expert DP | sampled token/logprobs 自洽 | 每步 sampled id 必须存在于对应 logprobs |
| MRV1 canary | DSv2 prefix cache | 检查末 token 竞态 | prefix on/off 输出按既有预期一致，无 token 0 污染 |
| MRV2 独立项 | DSpark Qwen3-8B | 检查 spec/rejection 路径 | 固定 seed 重复 3–5 次；若仍失败则单独定位，不归入 MRV1 Event |
| Structured output | xgrammar JSON | 检查 mask 与 sampled token 一致性 | 不出现 FSM reject；结果满足 grammar |
| LoRA | Llama3.2/iLLaMA TP2 | 覆盖 #47668 LoRA Event | 重复两次均与既有 reference 一致 |
| SP | Qwen3-VL SP TP2 | 覆盖 SP 输出读取 | SP on/off 符合既有精度标准，无非法首 token |
| CP | DSv2/Qwen3/DSv3.2/DSv4 4-card | 覆盖多卡时序 | 无重复/漏 token；满足现有 golden |
| Offload | prefetch eager + ACL graph | 覆盖 offloader Event | 不出现负数/越界 token；logprobs 自洽 |
| MoE residual | fused/native/custom selector | 验证 scaling exactly once | factor 不等于 1 时三条路径权重语义一致 |

## 7. 当前进展

- 已获取并分析 Actions run `29088445112` 的 9 个失败 job 完整日志。
- 已确认 24 个失败测试项及其共同的异步回传特征。
- 已确认 `(ba221520, ab7961a]` 范围以及 #47668/#47081 到下游 no-op Event 的代码因果链。
- 已确认 Transformers 本轮固定 5.5.4，不在当前精度修复中升级。
- 已识别 Hunyuan lazy registry、MoE scaling double-apply 和 CPU offload baseline helper 为独立问题。
- 已在本地 PR head 上完成 MRV1 Event-only 最小实现：`_torch_cuda_wrapper()` 退出后保持 `torch.cuda.Event = torch.npu.Event`，未修改 async 开关、golden、阈值或 Transformers 版本。
- 已创建本地修复提交 `b88020cb`（`Fix NPU event lifetime for async outputs`，含 Signed-off-by）：
  - CPU lifecycle UT 使用隔离的 fake torch 验证 wrapper 退出后仍保留 NPU Event，并透传 `blocking=True`；
  - A2 回归测试使用真实 NPU Event 执行 `record/synchronize/query`，防止再次退化为 placeholder。
- 本地验证已完成：Ruff 0.14.0 check/format、Python `py_compile`、`git diff --check` 均通过。
- 本交接记录随修复提交一次性发布；首轮新 CI 以发布后的 PR head 为准。本机无 NPU，硬件 A/B/精度验证由该轮 CI 完成。
- 记录快照时，run `29088445112` 中 A3 4-card part 2/3 与 part 3/3 曾处于运行中；后续观察必须以最新终态为准，不沿用旧状态推断。

## 8. 后续每小时 CI 观察项

每小时只记录变化；无变化时记录“无新增终态”，不要重复粘贴整份日志。

1. 记录时间、Actions run ID、下游 head SHA、verified vLLM SHA、实际 Transformers 版本。
2. 检查所有 job 状态，尤其 A3 4-card part 2/3、part 3/3 是否转为终态。
3. 新失败必须记录：job ID、测试全名、首个有效异常、最终 traceback、是否打印 async scheduling enabled。
4. 区分基础设施失败与测试失败：安装、下载、超时、设备不可用不得计入精度矩阵。
5. 对生成类失败提取：输出文本、token ids、对应 logprobs、prefix/LoRA/SP/CP/graph/spec 等关键配置。
6. 标记失败是否符合 Event 指纹：token 0/1、负数 token、sampled id 不在 logprobs、随机 `!`、重复/漏 token、grammar reject。
7. Event 修复提交出现后，优先比较同一用例修复前/修复后，而不是只比较总失败数。
8. 检查是否混入依赖变化；若 Transformers 不再是 5.5.4，该次结果不得直接与当前基线做精度归因。
9. acceptance 用例记录 seed、完整 acceptance 数组及重复次数；单次波动不作为修复结论。
10. 每次观察更新“已解决/仍失败/新增失败/基础设施失败”四类计数，并保留对应 job 链接。

## 9. 合入前清理检查

- [ ] Event、MoE、Hunyuan/Transformers 相关改动保持为独立提交或独立阶段。
- [ ] 没有为掩盖竞态而关闭 async scheduling。
- [ ] 没有更新 acceptance golden 或放宽精度阈值。
- [ ] 已完成至少一轮定向矩阵与一轮全量 main CI。
- [ ] 已将最终 RCA 和验证结果转移到正式 PR 描述、issue 或长期维护文档。
- [ ] **删除 `tools/main2main_0709_ci_handoff.md`，确认该临时文件不进入正式合入提交。**
