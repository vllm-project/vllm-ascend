# main2main 0709 CI 临时交接记录

> **临时文件：本文件仅用于 PR #11709 定位与交接，正式合入前必须删除，不得随功能提交进入主分支。**

## 1. 分析基线与已确认范围

- 下游 PR：`vllm-project/vllm-ascend#11709`，本轮残差分析基于远端 head `696323578a10e1f58c7c17c30a3e1e1ecdc715a1`；本地保留其上的 fixup 提交 `177eef18`。
- 本轮升级前全绿基线：下游 `1a56e8ebf6565661177de51936a7232f7d09b0b2`，vLLM main `ab7961a14a59be9a0170f1654315d5c2be44c015`，full E2E run `29140862370`。
- 当前目标 vLLM commit：`e5588e49bc2642670116664a7fc4096e27adb179`（2026-07-10 上午最后提交，`vllm#45261`）。
- `ab7961a` 是 `e5588e49` 的祖先；本轮新增适配区间为 `(ab7961a, e5588e49]`，共 28 个上游提交、95 个变更文件，其中 48 个位于 `vllm/`。
- 已知可以通过精度用例的 vLLM 边界：`ba22152096b2484faa3579624a253d54804d876d`。
- 前一轮已适配区间为 `(ba221520, ab7961a]`，共 155 个上游提交。
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
- 首次发布后的 run `29117216392` 只执行到 `markdownlint --fix`：上述二级列表被自动改为 4 空格缩进，因 hook 修改了工作树而失败；其余 pre-commit hook 全部通过，尚未进入硬件用例。
- 记录快照时，run `29088445112` 中 A3 4-card part 2/3 与 part 3/3 曾处于运行中；后续观察必须以最新终态为准，不沿用旧状态推断。

### 7.1 2026-07-11 最后两项失败的根因与修复

对比 PR #11510 全绿 run `29025366717` 与 PR #11709 run `29130818640` 后，最后两项失败已分别闭环；二者都不是通过修改 golden、放宽阈值或关闭 async scheduling 规避。

1. **MRV2 DSpark acceptance 确定性回归**
    - #11510 使用 vLLM `ba221520` 时通过；#11709 升级到 `ab7961a` 后，多次得到相同的低 acceptance 数组，因此不是随机精度波动。
    - 引入点是上游 `vllm#47914` / `0d12618e98ff2d21d36081e0e9b4eb23573b6d38`：`build_attn_metadata()` 的 `causal` 从单一 `bool` 扩展为按 KV cache group 配置的 `Mapping[int, bool]`。
    - DSpark 传入 `{0: False}` 表示第 0 组 non-causal；Ascend override 仍将整个非空 mapping 写入 `AscendCommonAttentionMetadata.causal`，后续布尔判断把它当成 `True`，错误选择 causal sparse mode。
    - 修复 `vllm_ascend/worker/v2/attn_utils.py`：逐 KV cache group 解析 `causal`，mapping 缺失的组按上游语义默认 `True`，传给 metadata 的始终是标量布尔值。
    - 新增 `tests/ut/worker/test_attn_utils.py`，验证 `{0: False}` 被解析为 `[False, True]`，同时覆盖显式 non-causal 与缺失组默认值。
2. **v0.23.0 hidden-state 文件读取竞态**
    - release anchor 始终是 vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`，commit 没有漂移；该失败可以受时序影响偶发通过，但不是 runner 基础设施错误。
    - 上游 `vllm#37374` / `4e2eba28beec9972445c338e8ad2080b3cab3246` 引入异步 D2H 与后台写盘，并提供带文件锁的 `load_hidden_states()` / `cleanup_hidden_states()` 消费协议。
    - 下游 `vllm-ascend#10459` / `801a6b41d29d23399bd8c9ebc2ea8883e2beefae` 为 v0.23.0 保留了直接 `exists + safe_open` 的读取分支，没有等待后台 writer；本 PR 的真实 NPU Event 修复 `b88020cb` 纠正 D2H 时序后，稳定暴露了既有写盘竞态。
    - 修复 `tests/e2e/pull_request/one_card/spec_decode/test_extract_hidden_states.py`：v0.23.0 与 main 统一使用 connector 的带锁加载和清理 helper，并用 `finally` 保证断言结束后清理文件；不增加 sleep 或轮询。

本地已完成 manual-stage pre-commit（含 Ruff、codespell、typos、markdownlint 和仓库自定义检查）及 `git diff --check`。本机 Python 未安装 torch 且没有 NPU，CPU 回归单测、硬件精度与竞态闭环仍由推送后的 CI 验收；v0.23.0 应重点连续复跑 `hybrid_dummy_eager`，main 应重点复跑 DSpark Qwen3-8B acceptance。

### 7.2 2026-07-11 v0.23.0 CPU UT 收集失败

- 推送 `66c2ebb0` 后，main CPU UT 已通过；v0.23.0 CPU job `86510109431` 在收集新增的 `tests/ut/worker/test_attn_utils.py` 时失败。
- traceback 为 `AttributeError: module 'vllm.v1.worker.gpu' has no attribute 'spec_decode'`。v0.23.0 虽包含 `gpu/spec_decode/speculator.py`，但本项目已明确禁用该版本的全部 V2 patch，release 只使用 V1，因此不会加载 V2 import 链。
- 这是 main-only V2 回归测试缺少版本门控，不是 DSpark causal 修复在 main 上失效。修复是在导入 `attn_utils` 前对 v0.23.0 执行模块级 skip；main lane 继续执行完整 `[False, True]` 回归断言，不扩大 v0.23.0 的生产支持范围。

### 7.3 历史备份记录：升级至 0710 上午 `e5588e49`

> 状态说明：本节和 7.4 记录回退前 `98fbb7dd` 的分析与实现，代码现保存在远程备份分支，不代表当前 `m2m-0709` 已携带这些适配。当前分支状态见 7.5。

升级前已确认 PR head `1a56e8eb` 的 full E2E run `29140862370` 全量通过。新目标 `e5588e49` 是 `ab7961a` 的线性后继；本轮只审计 `(ab7961a, e5588e49]` 的 28 个上游提交。main2main analyzer 给出 38 个下游候选文件，但结果仅作为检索提示，所有修改均经过上下游源码和运行门禁复核。

本轮确定需要适配的上游契约如下：

1. **`vllm#40996` / `95ed0feaa5`：DCP hybrid block-table 契约**
    - 上游 `_compute_slot_mapping_kernel` 新增必填 constexpr `KV_CACHE_BLOCK_SIZE` 和 `BLOCKS_PER_KV_BLOCK`，用于在物理 KV block 与 attention kernel block 不同时正确换算 table index。Ascend 直接调用该 kernel，main 若仍按旧签名启动会缺少必填参数；v0.23.0 的旧 kernel 又不接受新参数。
    - 上游新增 `KVCacheSpec.max_num_blocks_per_req()`，明确 Attention KV 按 DCP/PCP 分片、Mamba state 在各 rank 复制。Ascend 原逻辑先统一除 CP，再在 `BlockTable` 内单独乘回 Mamba，不能直接用于新 main，否则会重复缩放。
    - 同一改动把 worker 的 KV group 类型判断统一为 `get_kv_cache_spec_kind()`，使 `UniformTypeKVCacheSpecs` 包装的 Mamba/encoder-only group 仍能正确跳过 token slot mapping 或从 worker block-table 列表排除。worker 配置保留包装 spec，因此 Ascend 旧 direct `isinstance` 会把实际生产组误分类，并导致 block-size/kernel-size/max-row 三张表错位。
    - 310P 的受支持 hybrid/GDN 路径同样使用 Mamba block table；其 override 不会继承上游 `SlotMappingMode.NONE`，因此只同步 Mamba row-size 与跳过 slot mapping。310P attention backend 不支持 PCP/DCP，所以映射工具提示的 CP + split-block numpy 公式经复核后删除，不为非支持配置增加防御性语义。
    - 上游 `resolve_kv_cache_block_sizes()` 同步改为逐 group 计算 effective block size，并在 prefix cache **或 KV connector** 启用时计算 hash 粒度。Ascend 对该函数有整函数 monkeypatch，必须显式同步新语义，否则 hybrid + CP + connector 会得到与 scheduler/coordinator 不一致的 block/hash 粒度。
    - Ascend 整体覆写了 `HybridKVCacheCoordinator.find_longest_cache_hit` 的两条路径。上游把 EAGLE 额外探测步长从 `spec.block_size` 改为真实 `group_block_size`；若不跟随，DCP/压缩组会先少探测一块，随后 `drop_eagle_block` 再 pop，最终 prefix hit 少一整块。main 使用已计算的 `effective_block_size`，tag 保留旧步长。
    - 修复：所有新旧分支均由 `vllm_version_is("0.23.0")` 决定；tag 保留旧 kernel kwargs 和旧 row-size/哈希算法，main 使用新 constexpr、新 spec helper，并仅对 Attention group 应用 CP 缩放。coordinator 在 scheduler 已解包后的实际 Mamba spec 上同步 effective block size 和 EAGLE 探测步长；没有参数探测、异常兜底或不可达 wrapper 分支。
2. **`vllm#47381` / `85b3a7264b`：MRV2 请求排序与 DeepSeek MTP top-k**
    - 上游 MRV2 不再简单按 scheduled-token 数排序，而是用 `sort_batch_req_ids(..., decode_query_len)` 将 uniform decode 放在 short-extend 前；Ascend 覆盖的 `prepare_inputs()` 仍是旧排序，会使 `split_decodes_and_prefills` 把 spec decode 错判为 prefill。
    - 同一 PR 把 DeepSeek MTP layer 的初始 `skip_topk` 强制为 false，避免 draft step 读取从未写入的 top-k buffer；Ascend 整体替换 `DeepseekV2MLAAttention.__init__`，因此不会继承上游修复。
    - 修复：main 调用上游排序 helper，并对 MTP 应用新 `skip_topk` 语义；tag 由 `vllm_version_is` 保留原行为。
3. **`vllm#46865` / `2285cfca46`：MultiConnector 分配后调用协议**
    - 新 main 要求所有 sub-connector 接收请求的真实 blocks，只以 `num_external_tokens=0` 表示“未被选中、不加载”。Ascend override 仍向未选中的 connector 传 empty blocks，并通过 Layerwise 特判绕开旧协议。
    - 修复：main 对普通未选中 connector 按新协议传真实 blocks 和 0 external tokens，且不再创建无用 empty blocks；tag 继续创建并传 empty blocks。下游 Layerwise + KV-pool 组合仍保留其既有特判：即使由另一 connector 命中，Layerwise 也必须看到真实 blocks 和该次 external-token 状态，否则会以空 blocks 提前消费 `do_remote_prefill`。这是下游 PR #7032 已有的局部例外，不在本轮扩大或删除。
4. **`vllm#46694` / `ae6170f874`：P/D async KV load 的 lookahead**
    - 上游把限制条件从 `load_kv_async and use_eagle` 扩为 `load_kv_async and num_lookahead_tokens > 0`：async load 当步不执行 forward，MTP 等其他 speculative method 同样不能提前分配 lookahead blocks。
    - 普通 scheduler 会继承上游；三个自研 scheduler 只在 `request.num_computed_tokens == 0` 查询 async load，既有逻辑已在同一条件把 lookahead 置零。`BalanceScheduler.schedule()` 是 v0.23 整体复制体，必须单独同步：通过 `vllm_version_is` 让 tag 保留 Eagle 条件、main 使用所有 speculative method 的新条件。
5. **`vllm#48085` / `1cd75b3dd4`：KVBlockZeroer pinned-buffer 竞态**
    - 非阻塞 pinned-host→device copy 完成前复用同一个 pinned ID buffer，会让下一批覆盖上一批仍在传输的 block IDs，最终清零错误 KV block。Ascend 复制了同样的单 buffer 实现；310P 使用同步 tensor zero，不受此竞态影响。
    - 修复：按 `max_concurrent_batches` 分配并轮转 pinned/GPU ID buffer；扩容前同步，避免释放仍在传输的源 buffer。该修改的触发依据是 main 新引入的上游修复被 Ascend 整体 override 遮蔽，不是映射同名；tag 使用相同的 nonblocking 单-buffer 实现并具备多 in-flight batch，因此共用确定性修复，明确作为双 lane bugfix，而不是实验性兜底。

逐项关系复核结果：

| 上游 PR | 下游实际关系 | 不跟随的具体后果 | 结论 |
| --- | --- | --- | --- |
| #40996 | direct kernel call；MRV1/310P runner override；KV cache utils/coordinator monkeypatch | main kernel 缺 constexpr；Mamba row/slot mode 错误；hybrid hash 粒度不一致；EAGLE prefix hit 少一块 | 仅保留生产可达修改，并补齐 EAGLE 两条路径 |
| #47381 | MRV2 `prepare_inputs` override；DeepSeek 构造器直接替换 | spec decode 被误分为 prefill；MTP 读取未写 top-k buffer | 保留，main/tag 由 `vllm_version_is` 分流 |
| #46865 | 工厂移除上游注册并注册 `AscendMultiConnector`；方法 override | 非选中 connector 丢失真实 block bookkeeping | 保留普通 connector 新协议；保留下游 Layerwise 既有例外 |
| #46694 | `BalanceScheduler.schedule` 为 tag 整体复制体并替换上游 Scheduler | 非 Eagle speculative async load 提前分配 lookahead，导致本地/远端 block 数不一致 | 仅修改 BalanceScheduler；其他 scheduler 或继承路径不复制 |
| #48085 | `AscendKVBlockZeroer` 覆写 ctor/init/zero 全路径 | 下一批覆盖在途 pinned IDs，清零错误 KV block | 保留 main 对齐；同一确定性竞态双 lane 修复 |

已人工判定不修改的 analyzer/邻近候选：

- 310P CP + split-block numpy slot mapping 只在不受支持的 PCP/DCP 配置或合成 UT 中可达；310P backend 没有 CP attention 实现，因此删除该防御性公式，只保留 CP=1 可达的 Mamba 修改。
- scheduler coordinator 接收的配置已由上游 `generate_scheduler_kv_cache_config()` 解包 `UniformTypeKVCacheSpecs`；因此删除 coordinator 的 Uniform-Mamba 分类和人工构造 UT。worker 侧配置不会解包，相关 row-size/slot-mode 适配继续保留。
- `vllm#48135` / `e08a9151` 的 Tensor `causal` 只修复 DiffusionGemma；Ascend 当前无该模型支持，也没有可消费 per-request Tensor causal 的 attention backend。仅扩类型会把错误从 `.get()` 推迟到布尔判断，不构成完整适配，因此不改。
- `vllm#47317` / `2ded1b24e7ef48e21f71f51fc923a352deefbde2` 只减少 Mooncake SWA lookup 中被 mask chunk 的 hash 分配，PR 明确说明输出不变，属于性能优化而非升级兼容缺口。
- `vllm#48132` 的 Mamba state reset 位于上游 model-state selector，但 Ascend V2 override 绕过该实现；当前 Ascend 不支持 hybrid Mamba + MRV2 speculative decode，也没有可触发该契约的消费者，因此本 PR 不引入无运行路径的修改。`vllm#47923`、`vllm#45261` 的 KV offload/full-report event 改动由未覆盖的上游实现直接生效；ROCm/XPU/CI/已删除模型提交不属于 Ascend main2main 适配范围。

本节记录的是推送前源码闭包；推送后只处理新 CI 中能够证明由 `(ab7961a, e5588e49]` 引入、且属于上述或新增上游契约变化的失败。不得更新 golden、放宽阈值、关闭功能或增加兜底来换取用例通过。

### 7.4 历史备份复核：0710 适配必要性与测试范围

- 重新运行 main2main evaluator 后，38 个预测文件仅 8 个与实际修改相交，precision 为 0.211；因此所有候选重新按 direct call、override、monkeypatch、注册替换及实际运行门禁逐项验证。复核删除了 310P 非支持 CP 公式与 coordinator 不可达 Uniform-Mamba 分支，并发现、补齐了工具未直接给出结论的两处 EAGLE effective-block 漏同步。
- 非 #40996 的四组生产修改也已逐项复核：#47381 两处均为真实 override；#46865 为工厂注册替换；#46694 只影响复制版 BalanceScheduler；#48085 为 Ascend 全路径覆写。未因同名接口、参数存在性或测试期望而新增生产修改。
- 对 `(1a56e8eb, 0710 head]` 的测试改动按 PR 职责再次精简：纯新增回归看护用例不在当前 main2main PR 中呈现，完整版本已推送到远程分支 `codex/m2m-0710-adaptation-tests`，仅作备份，不创建 PR。
- 本 PR 只保留既有 UT 为适配新接口所必需的调整：310P runner fixture 补齐 `VllmConfig` 字段；BalanceScheduler 既有 tag-copy AST 检查规范化版本 helper；hybrid CP block-size 既有断言按 main/tag 契约分流。
- 精简前生产 head `d0e5afc4` 的 run `29147491841` 已确认远程 pre-commit、mypy、coverage config、test selection 以及 main `e5588e49` / tag `v0.23.0` 双 CPU UT 全部成功；精简后的最终结果以新 head CI 为准。

### 7.5 2026-07-11 0710 重新基线化与跨设备交接

#### 已确认全绿的基线

- 下游 `1a56e8ebf6565661177de51936a7232f7d09b0b2` 对应 vLLM main `ab7961a14a59be9a0170f1654315d5c2be44c015`，full E2E run `29140862370` 已全量通过：30 个实际 job 成功，2 个条件性 coverage 上传 job 跳过，无失败或取消。
- 该基线已经包含并验证此前 0709 main2main 所需修复：
    - MRV1 运行期 `torch.cuda.Event` 保持真实 NPU Event，修复异步输出读取竞态；
    - 在 Transformers `5.5.4` 基线下补齐 HunyuanVL processor 与 lazy registry 兼容；
    - KV cache admission 使用 `max_in_flight_tokens`，并传递 KV cache zeroing 需求；
    - 按 KV cache group 解析 DSpark `causal`，修复 MRV2 acceptance 回归；
    - hidden-state E2E 使用 connector 带锁加载和清理协议，消除后台写盘竞态；
    - 对 v0.23.0 门控 main-only worker V2 UT，修复 release lane 收集失败。
- 本轮从该已验证节点重新升级，避免将映射工具提示、预防性修复和真实 CI 必需修改混入同一轮验证。

#### 两个远程备份分支

| 分支 | 快照 | 保存内容 | 状态 |
| --- | --- | --- | --- |
| `codex/m2m-0709-full-backup-20260711` | `98fbb7dd7eac90516428e75f05f60b26486370bf` | 回退前完整 `m2m-0709`；包含目标 `e5588e49`、#40996、#47381、#46865、#46694、#48085 等 0710 生产适配及精简后的必要 UT 调整 | 已推送，仅备份，不创建 PR |
| `codex/m2m-0710-adaptation-tests` | `8e8e4e35f52a845ab4f0f05ef0808d065e2d0177` | 上述生产适配精简测试前的完整版本；保留 BlockTable、KVBlockZeroer、MultiConnector、Event、`attn_utils`、Hunyuan 等防护性回归测试 | 已推送，仅备份，不创建 PR |

两个分支均不得删除或强推覆盖。后续恢复适配时，应按上游 PR 和具体修改点逐项审查、单独 cherry-pick，不要整体合回。

#### 当前 `m2m-0709` 的重新起点

- `m2m-0709` 已回到全绿节点 `1a56e8eb`，并由独立提交 `39b657be` 仅将 `.github/vllm-main-verified.commit` 更新为 `e5588e49bc2642670116664a7fc4096e27adb179`。
- 除本交接文档外，当前分支没有预先带回 `(ab7961a, e5588e49]` 的生产适配或新增测试。
- 首轮 CI 的目的，是直接暴露升级导致的编译、收集、UT、E2E 和精度问题，而不是证明备份分支中的全部修改都必要。

#### CI 优先策略

1. 优先保证 PR 全量 CI 通过并具备合入条件。
2. 新失败先确认是否由 `(ab7961a, e5588e49]` 的上游变更引入；非升级适配问题不在本 PR 修复。
3. 每个修复记录失败 job、首个有效异常、上游 PR/commit、上游具体修改点，以及下游 direct call、override、monkeypatch、继承或注册替换关系。
4. main/tag 差异使用 `vllm_version_is(...)` 明确分流，不使用参数探测、异常兜底、静默 fallback、关闭功能、更新 golden 或放宽阈值。
5. main2main 映射结果只作为检索提示；没有真实上下游关系或运行路径的候选不修改。
6. 首轮 CI 全量通过后，再从两个备份分支审计 CI 未覆盖但确定存在的适配缺口；这些补充修改另行提交和验证，不阻塞当前 PR 优先合入。

#### 下一台机器的继续步骤

1. `git fetch` 后确认远程 `m2m-0709`、`codex/m2m-0709-full-backup-20260711` 和 `codex/m2m-0710-adaptation-tests` 均存在，并记录各自 SHA。
2. 确认 `m2m-0709` 包含全绿祖先 `1a56e8eb`，且 `.github/vllm-main-verified.commit` 精确指向 `e5588e49bc2642670116664a7fc4096e27adb179`。
3. 查询新 Actions run，记录 run ID、PR head SHA、main/tag 实际安装版本及失败矩阵。
4. 按“CI 失败 → 上游根因 → 下游真实覆盖关系 → 最小修复”的顺序处理；每轮只引入能够由失败证据证明必要的修改。
5. CI 全量通过后先稳定当前可合入版本；随后再审计并分批吸收备份分支中的非 CI 覆盖适配。

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
