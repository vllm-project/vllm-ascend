# MoE Fused Baseline

本文档总结 `vllm_ascend/ops/fused_moe` 的当前实现能力，作为 `vllm-ascend-model-adapter` skill 在适配新 MoE 模型时的 **MoE 能力基线**。

目标不是解释通用 MoE 原理，而是回答这几个适配问题：

1. 当前 `vllm-ascend` 的 MoE 路径已经支持什么。
2. 当前实现对新模型的 MoE 层有哪些结构假设。
3. 新模型的 router / experts / shared expert / EP 路径和现有基线差在哪里。
4. 差异更可能落在 upstream vLLM 模型适配、权重映射、框架接线，还是 Ascend backend 能力本身。

## 1. 代码范围

主要分析对象：

- `vllm_ascend/ops/fused_moe/fused_moe.py`
- `vllm_ascend/ops/fused_moe/experts_selector.py`
- `vllm_ascend/ops/fused_moe/moe_comm_method.py`
- `vllm_ascend/ops/fused_moe/token_dispatcher.py`
- `vllm_ascend/ops/fused_moe/prepare_finalize.py`
- `vllm_ascend/ops/fused_moe/moe_mlp.py`
- `vllm_ascend/ops/fused_moe/moe_runtime_args.py`

相关依赖：

- `vllm_ascend/utils.py`
- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/ascend_forward_context.py`

## 2. 总体结论

当前 Ascend MoE 路径不是“单一 fused op”，而是一条分层流水线：

1. `prepare`
2. `router/top-k select`
3. `token dispatch`
4. `expert MLP`
5. `token combine`
6. `finalize`

其中最核心的事实有三点：

1. 当前实现已经覆盖了 **decoder-only routed MoE LLM** 的主流执行路径，尤其是 `topk router + SwiGLU expert + down_proj` 这一类结构。
2. 当前实现强依赖一个 **两段式 expert MLP 假设**：第一段是 `w13`/`gate_up`，中间是 `swiglu`，第二段是 `w2`/`down`。
3. 适配新模型时，最常见的问题通常不是 “Ascend 没有 MoE”，而是：
   - router 规则不同；
   - expert 权重布局不同；
   - shared expert 接线不同；
   - EP/TP 路径假设不同；
   - quant / comm / metadata 契约没有接上。

## 3. 实现入口和主流水线

### 3.1 Layer 入口

`AscendFusedMoE` 在 [fused_moe.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py:93) 这一组类里接管 upstream `FusedMoE` 行为。

关键点：

- `AscendUnquantizedFusedMoEMethod.is_monolithic=False`
- `maybe_make_prepare_finalize()` 直接返回 `None`
- `AscendMoERunner.forward_impl()` 改走 Ascend 自己的 `forward_impl`

这表示当前实现 **明确绕开 upstream modular-kernel prepare/finalize 初始化**，使用 Ascend 自己的通信与执行栈。

### 3.2 流水线中枢

真正的流水线中枢在 [moe_comm_method.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py:122)：

1. `build_token_dispatch_input`
2. `token_dispatcher.token_dispatch(...)`
3. `build_mlp_compute_input`
4. `unified_apply_mlp(...)`
5. `token_dispatcher.token_combine(...)`

也就是说，当前 MoE 能力可以按 4 个问题拆开分析：

- router 怎么选专家；
- token 怎么发给专家；
- expert 内部怎么算；
- 结果怎么合并回来。

这对 skill 很重要，因为新模型的适配差异通常也恰好落在这四块之一。

## 4. 当前支持的 MoE 结构假设

### 4.1 已有结构假设

从 [fused_moe.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py:108) 和 [moe_mlp.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py:344) 看，当前实现默认专家结构是：

1. `router_logits -> topk_ids/topk_weights`
2. `expert gate_up / w13`
3. `swiglu` 或兼容激活
4. `expert down / w2`

权重上默认围绕这些对象组织：

- `layer.w13_weight`
- `layer.w2_weight`
- 可选 `w13_bias`
- 可选 `w2_bias`
- 可选 scale / offset / scale_bias

因此，这份基线最适合拿来对比下面这类模型：

- DeepSeek 风格 routed experts
- Qwen / Mixtral 一类的 `gate/up/down` expert 结构
- 共享同一 router 但每个 expert 仍是两段式 MLP 的模型

### 4.2 当前不应直接假定已支持的结构

以下情况不能只看“仓库里有 MoE”就认为可直接跑通：

1. expert 不是 `gate_up + swiglu + down` 两段式。
2. expert 内部带额外线性层、卷积、并行支路或 residual expert。
3. router 不是标准 top-k gating，或者需要特殊归一化/后处理。
4. shared expert 不是当前实现假设的并行拼接方式。
5. expert 权重命名和布局与现有 loader 完全不同。

这些都应先判定为 **MoE layer gap**，再决定是 adapter 问题还是 backend 问题。

## 5. Router / Gate 能力基线

### 5.1 主入口

router 入口在 [experts_selector.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/experts_selector.py:30) 的 `select_experts(...)`。

它先做能力判定：

- 能用 NPU fused gating 就走 fused path
- 否则回退到 native path

### 5.2 已适配的 router 特性

从 [check_npu_moe_gating_top_k](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/experts_selector.py:140) 和 `_select_experts_with_fusion_ops(...)` 看，当前已有能力包括：

- `softmax` top-k
- `sigmoid` top-k
- `sqrtsoftplus` top-k
- grouped top-k
- `renormalize`
- `e_score_correction_bias`
- `routed_scaling_factor`
- 使用 `tid2eid + input_ids` 的 hash routing 特化路径
- `mix_placement` 下给 shared expert 补 expert id / weight

### 5.3 典型 fused path

当前主要有两类 fused gate 路径：

1. `DeviceOperator.moe_gating_top_k(...)`
2. `torch.ops._C_ascend.moe_gating_top_k_hash(...)`

其中第二类更像 DeepSeek V4 / hash routing 的特化支持。

### 5.4 fallback 条件

以下情况会使代码回退到 native router 路径：

- `custom_routing_function` 不为空
- `scoring_func` 不在当前支持集合内
- `sigmoid` 且 `renormalize=False`
- 分组参数不满足当前 fused op 约束

这说明 skill 在看新模型时，不能只看 “top_k=8”。还要看：

- scoring function
- grouped top-k 组织方式
- correction bias
- 是否自定义 routing 函数
- 是否依赖 token id hash

## 6. Token Dispatch / Combine 能力基线

### 6.1 当前支持的通信类型

[setup_moe_comm_method](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py:55) 当前注册：

- `ALLGATHER`
- `ALLTOALL`
- `MC2`
- `FUSED_MC2`

当 `ep_size == 1` 时，默认只建 `ALLGATHER`。

### 6.2 各通信类型的定位

| 通信类型 | 当前定位 | 典型依赖 | 适配含义 |
| --- | --- | --- | --- |
| `ALLGATHER` | 通用默认路径 | `npu_moe_init_routing` + `npu_moe_token_unpermute` | 兼容性最好，优先判断新模型是否能先接上这条路 |
| `ALLTOALL` | EP 场景更优性能路径 | all-to-all-v + grouped matmul | 模型层面通常不需要感知，但 EP 行为要能对齐 |
| `MC2` | Ascend 特化通信计算并行路径 | `npu_moe_distribute_dispatch/combine` | 依赖更多 forward context 和分布式约束 |
| `FUSED_MC2` | 更激进的 dispatch+ffn+combine 融合路径 | `_C_ascend.dispatch_ffn_combine` / `dispatch_gmm_combine_decode` | 权重格式和入参契约最强 |

### 6.3 AllGather 路径

[AllGatherCommImpl](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py:185) 是当前最通用的基线路径。

特点：

- 默认兼容性最好
- `token_dispatch` 依赖 `npu_moe_init_routing`
- `token_combine` 依赖 `npu_moe_token_unpermute`
- 某些 quant 路径可直接在 dispatch 前后带量化信息
- `apply_router_weight_on_input` 只支持 `topk=1`

对 skill 的意义是：

- 如果新模型只是 expert 命名或 router 接线不同，通常应先争取接到 AllGather 路径验证。
- 如果连 AllGather 都接不上，再考虑是不是 backend 结构假设不匹配。

### 6.4 MC2 路径

[TokenDispatcherWithMC2](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/token_dispatcher.py:101) 是当前最重要的 Ascend 特化分发器之一。

核心特性：

- 调用 `torch_npu.npu_moe_distribute_dispatch[_v2]`
- combine 调用 `torch_npu.npu_moe_distribute_combine[_v2]`
- 支持 `expert_map`
- 支持 `global_redundant_expert_num`
- 支持 `mc2_mask`
- 支持 hierarchy comm
- 支持 comm quant mode
- 当 `should_skip_allreduce_across_dp_group(...)` 为真时，`global_bs` 取真实上界且不再传 `mc2_mask`

这意味着 MC2 路径依赖的不是单个 layer，而是更完整的 runtime 条件：

- forward context 中的 `mc2_mask`
- 跨 DP 的 token 对齐策略
- EP/TP group 设置
- 分布式 world size 与 max token 估计

### 6.5 Fused MC2 路径

[FusedMC2CommImpl](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py:259) 进一步把 dispatch / ffn / combine 融到 custom op：

- `dispatch_ffn_combine`
- `dispatch_gmm_combine_decode`

其特征是：

- 对输入签名要求更强
- `w1` / `w2` 要按 list 形式传
- 浮点场景仍要传 dummy scale tensor
- 依赖 `expert_map`
- 更依赖 NZ 权重格式

因此，Fused MC2 更像“性能增强路径”，而不是“新模型 bring-up 的第一落点”。

## 7. Prepare / Finalize 能力基线

### 7.1 作用

[prepare_finalize.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/prepare_finalize.py:40) 负责：

- padding
- TP 切分
- DP/EP gather
- finalize 时还原形状
- 在必要时带上量化所需的 per-token scale

### 7.2 All2All / MC2

`PrepareAndFinalizeWithAll2All` 和 `PrepareAndFinalizeWithMC2` 的共同特点：

- 在 token 维度做 padding / split
- finalize 时恢复 token 排布
- MC2 额外依赖 `_EXTRA_CTX.mc2_mask`
- MC2 额外依赖 `_EXTRA_CTX.padded_num_tokens`

这意味着任何新模型如果改变了：

- batch token 组织方式
- speculative / splitfuse 下 token 排列
- shared expert DP 行为

都可能不是 router 本身的问题，而是 prepare/finalize 契约被破坏。

### 7.3 AllGather

`PrepareAndFinalizeWithAllGather` 兼容两类场景：

1. 非 SP 时基于 DP 组处理
2. 开启 SP 时走 EP 组处理

并且可在 prepare 阶段把 hidden states 预量化到：

- `W8A8`
- `MXFP8`

这说明 MoE 量化路径不只存在于 MLP 核心算子里，也存在于 prepare 阶段的数据进入方式里。

## 8. Expert MLP 能力基线

### 8.1 主结构

[unified_apply_mlp](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py:344) 最核心的结构假设是：

1. `gmm1` 计算 `gate_up`
2. 激活一般是 `swiglu`
3. `gmm2` 计算 `down`

这是当前 MoE baseline 最强的结构前提。

### 8.2 非量化路径

非量化路径主要围绕：

- `w1`
- `w2`
- `group_list`
- `group_list_type`
- `activation`

默认 expert 权重是按 grouped matmul 组织的，并且常见情况下要先转置。

### 8.3 量化路径

当前 quant MoE MLP 已覆盖多条路径：

- `W8A8`
- `W4A8`
- `MXFP8`
- `MXFP4`
- per-channel W4A8 特化
- antiquant offset 路径
- 自定义 fused grouped matmul + swiglu quant 路径

其中 `group_list` / `group_list_type` 是强约束参数，因为不同内核要求：

- prefix sum
- per-expert count
- 或其他中间编码

### 8.4 模型适配含义

如果新模型的 expert 结构不是当前两段式 MLP，那么问题通常不该直接归类成“量化不支持”，而是：

- expert 结构 baseline 不匹配；
- 现有 grouped matmul / swiglu contract 不成立；
- 需要新的 upstream MoE layer 建模，甚至新的 Ascend backend 支持。

## 9. 权重布局和格式假设

### 9.1 权重后处理

[process_weights_after_loading](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py:108) 做了几件关键事：

- pad 权重
- `w13_weight.transpose(1, 2).contiguous()`
- `w2_weight.transpose(1, 2).contiguous()`
- `enable_fused_mc2` 时强制转 `FRACTAL_NZ`
- 否则走 `maybe_trans_nz(...)`

这说明当前 MoE backend 对权重布局有明确要求，而不是接受任意原始 checkpoint 布局。

### 9.2 适配判断

如果新模型：

- expert 权重不是 `w13` / `w2`
- gate 和 up 不是合并权重
- down 权重维度顺序不同
- shared expert 权重单独打包
- checkpoint 是别的 tensor packing 方式

优先怀疑：

1. upstream vLLM 模型适配层没有正确把权重接成当前 MoE contract；
2. 当前 baseline 的 expert weight layout 与模型要求不一致。

不要第一反应就改 backend。

## 10. Shared Expert / Mix Placement / Dynamic EPLB

### 10.1 Shared expert

从 [fused_moe.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py:157) 及后续 `forward_impl` 逻辑可见，当前实现已经考虑：

- `n_shared_experts`
- shared expert 独立前向
- shared expert 与 routed expert 并行计算
- shared expert 与 routed output 的 reduce 行为协调

因此，shared expert 不是完全空白能力。

但需要注意：

- 当前 shared expert 是建立在现有 `AscendFusedMoE` 组织方式上的。
- 如果新模型的 shared expert 是 residual MLP、串联 MoE、或权重共享方式特殊，仍需要单独做 gap analysis。

### 10.2 Mix placement

router 阶段已经支持 `mix_placement`：

- 通过给 `topk_ids/topk_weights` 拼接 shared expert 条目完成

这说明“部分 expert routed，部分 expert 常驻”这类混合放置不是完全没有能力。

### 10.3 Dynamic EPLB

当前 MoE 支持 dynamic expert load balance：

- 初始化 `eplb_config`
- forward 结束后更新 `moe_load`
- routing / mlp 路径都带 `dynamic_eplb` 参数

因此，如果新模型 MoE 带 expert remap / redundant expert / load balance 元数据，要先看是否能映射到：

- `expert_map`
- `global_redundant_expert_num`
- `log2phy`

而不是直接认为“不支持 expert balance”。

## 11. 运行时契约

当前 fused MoE runtime contract 已经在 [moe_runtime_args.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/fused_moe/moe_runtime_args.py:17) 统一。

适配时最应关注这些对象：

### 11.1 Routing contract

- `expert_map`
- `global_redundant_expert_num`
- `mc2_mask`
- `apply_router_weight_on_input`
- `log2phy`
- `pertoken_scale`

### 11.2 Weight contract

- `w1`
- `w2`
- `w1_bias`
- `w2_bias`
- `w1_scale`
- `w2_scale`
- `w1_scale_bias`
- `w2_scale_bias`
- `w1_offset`
- `w2_offset`

### 11.3 Quant contract

- `quant_type`
- `comm_quant_mode`
- MXFP 参数
- `is_per_channel_weight`

skill 在分析新模型 MoE 层时，应尽量把模型需求翻译成这组 contract，看哪些字段已有映射、哪些字段没有来源。

## 12. 与 model runner 的耦合点

MoE 支持并不只在 `fused_moe/` 目录内部。

### 12.1 MoE 检测

[utils.py](/home/cmq/code/vllm-ascend/vllm_ascend/utils.py:203) 附近的辅助逻辑表明，框架会通过 config 内容递归判断模型是否带 expert。

这意味着如果新模型 config 的 MoE 描述很特殊，第一层问题可能就是“没有被正确判定为 MoE 模型”。

### 12.2 DP allreduce skip 逻辑

`should_skip_allreduce_across_dp_group(...)` 会根据：

- 是否是 MoE
- 是否启用 hierarchy comm
- 是否是 draft model

来决定 DP 维度上的行为。

这直接影响 MC2 token dispatch 的 `global_bs` / `mc2_mask` 分支。

### 12.3 model runner 侧行为

[model_runner_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:568) 之后这段逻辑说明：

- token 数量需要跨 DP 对齐
- spec decode / graph capture / mixed batch 会影响 metadata
- MoE 运行时还依赖输入 token 对齐和 forward context 中的额外字段

另外，当前实现还会单独处理 routed experts capture，这说明 Ascend MoE 已经和上游默认 hook 有一定偏离。

## 13. 当前已覆盖的典型场景

基于现有代码，可以认为当前已经覆盖或部分覆盖这些场景：

1. decoder-only routed MoE LLM
2. top-k router
3. grouped top-k
4. `softmax` / `sigmoid` / `sqrtsoftplus`
5. `e_score_correction_bias`
6. hash routing 特化路径
7. shared expert
8. mix placement
9. dynamic EPLB
10. `ALLGATHER` / `ALLTOALL` / `MC2` / `FUSED_MC2`
11. EP 路径
12. W8A8 / W4A8 / MXFP8 / MXFP4 MoE 量化路径

但“覆盖”不等于“任何 MoE 架构都直接兼容”。当前覆盖更准确地说是：

**已经覆盖一组特定结构和特定 contract 下的 Ascend MoE 实现。**

## 14. 当前高风险差异点

新模型如果命中以下差异点，通常要优先做 MoE gap analysis：

1. router 不是标准 top-k
2. grouped top-k 规则不同
3. router 需要 custom routing function
4. expert 不是两段式 `gate_up + swiglu + down`
5. gate/up 权重不是合并形式
6. shared expert 结构不同
7. checkpoint expert 权重布局和现有 loader 假设不一致
8. 依赖新的 communication metadata
9. 依赖新的 comm quant 模式
10. 运行时 token 组织方式与当前 prepare/finalize 假设不一致

## 15. 适配时的优先排查顺序

对于新 MoE 模型，建议按这个顺序判断：

1. `config.json` 是否明确暴露 expert/router/shared-expert 结构。
2. modeling code 是否仍然能映射到当前 `router + experts + shared expert` 三段组织。
3. expert 是否能映射到 `w13/w2 + swiglu` contract。
4. router 是否能映射到当前 `select_experts(...)` 能力集合。
5. 模型是否应先走 `ALLGATHER` 基线验证，而不是直接追 MC2/FusedMC2。
6. quant checkpoint 是否能映射到当前 quant contract。
7. 若以上都成立，再判断 EP / MC2 / fused 性能路径是否需要补适配。

## 16. Skill 使用要求

当 skill 适配 `moe llm` 时，应把这份文档当作 **当前 MoE capability baseline**，并在读取新模型的 `config.json`、modeling code、权重键和运行现象后，写出固定格式的小节：

```markdown
## MoE Gap Analysis

### 1. Current Capability
- Router capability baseline:
- Expert MLP baseline:
- Shared expert baseline:
- Communication baseline:
- Quantization baseline:

### 2. Model Requirement
- Router/gate behavior:
- Expert structure:
- Shared expert / residual MLP behavior:
- EP/TP/dispatch expectations:
- Quant / weight-layout requirements:

### 3. Gap
- Router gap:
- Expert-structure gap:
- Weight-layout gap:
- Communication/runtime-contract gap:
- Unknowns to verify:

### 4. Adaptation Plan
- Fix location:
- Minimal files to touch:
- First validation path:
- Stop / escalate condition:
```

这个小节的作用是先判断：

- 新模型 MoE 层是不是其实已经能接到现有路径；
- 还是需要改 upstream vLLM 的模型 adapter / weight loader；
- 还是需要改 vLLM 框架接线；
- 还是 Ascend backend 本身没有对应能力。

## 17. 对 skill 的直接指导结论

对于 skill 来说，当前应采用下面的判断口径：

1. 把 `fused_moe` 看作 **已有能力基线**，不是空白区域。
2. 新模型 MoE 适配默认先比对：
   - router 规则
   - expert MLP 结构
   - shared expert 结构
   - weight layout
   - EP/MC2/runtime contract
3. 优先把问题分类成：
   - 模型建模/注册问题
   - 权重映射问题
   - 现有 MoE contract 接线问题
   - backend 能力缺口
4. bring-up 时优先争取先接通 `ALLGATHER` 基线路径，再决定是否追 EP/MC2/FusedMC2。

如果一个新模型的 MoE 层能映射到当前基线，适配重点通常是 **upstream vLLM 侧的模型接线和权重映射**，而不是从零新写 Ascend MoE backend。
