# Kimi K3 A5 MegaMoE 适配设计

## 1. 文档信息

| 项目 | 内容 |
|---|---|
| 目标代码基线 | vLLM Ascend `releases/v0.26.0rc`，`f1726dff1` |
| 目标模型 | Kimi K3，Latent MoE，SiTU，MXFP4 权重 / MXFP8 激活 |
| 目标硬件 | Ascend A5，首期验证 4 节点 32 卡，EP=32 |
| 参考实现 | [PR #14449: Add A5 MegaMoE fused backend](https://github.com/vllm-project/vllm-ascend/pull/14449)，vLLM 0.23，已完成 DSV4 模型验证 |
| 算子参考 | `cann-recipes-infer` Kimi K3，`2baf6f14c` |
| 文档状态 | 已按方案实现，待 A5 32 卡精度与性能验收 |

面向代码使用者的精简说明见
[Kimi K3 A5 MegaMoE 算子替换说明](kimi_k3_a5_megamoe_replacement.md)。本文保留完整设计、
历史取舍、风险和分阶段验收依据。

## 2. 摘要

本方案在 vLLM Ascend 0.26 的通用 `FusedMoE` 框架中增加 A5 MegaMoE 后端，用一个融合算子替换 Kimi K3 路由专家内部的 Dispatch、GMM1、SiTU、GMM2 和 Combine。模型外层的 Gate、latent down projection、routed RMSNorm、latent up projection 和共享专家保持不变。

```text
full hidden [T, 7168]
    |-- gate ------------------------------------------> top-k ids / weights
    |-- routed_expert_down_proj ----------------------> latent hidden [T, 3584]
                                                           |
                                                           v
                                      +----------------------------------------+
                                      | MegaMoE                               |
                                      | dispatch -> GMM1 -> SiTU -> GMM2       |
                                      |          -> weighted combine           |
                                      +----------------------------------------+
                                                           |
                                                           v
                                                   [T, 3584], globally combined
                                                           |
                                     routed RMSNorm -> routed up projection
                                                           |
shared expert(full hidden) ------------------------------- + -> [T, 7168]
```

核心结论如下：

1. PR #14449 的 A5 backend、独立通信域、DP 对齐和 buffer 生命周期设计可以复用。
2. PR #14449 的最终实现只支持 SwiGLU，不能直接用于 K3。K3 必须增加 `situglu` 和 `activation_params` 传递。
3. 不能整体 cherry-pick 0.23 PR。0.26 已重构 `FusedMoE`、运行参数和 K3 routed transform，需要按当前接口迁移。
4. 首期只支持运行时解析为 `W4A8MXFP` 的 MXFP4/MXFP8 契约。现有 K3 文档使用 ModelSlim W4A8 checkpoint，必须以实际 `QuantType` 判断，不能根据模型名称进入该路径。
5. MegaMoE 是全组 collective。所有 rank 必须选择同一后端、使用相同 token shape，并共享同一套不可变 symmetric buffer 配置。

## 3. 背景与参考实现结论

### 3.1 PR #14449 提供的基线

PR #14449 基于 vLLM 0.23，在 A5 上将支持的 MXFP MoE 请求映射到逻辑 `FUSED_MC2` 路径，并通过 `MegaMoEBackend` 一次完成 dispatch、专家计算和 combine。该方案已在 DSV4 上完成模型验证，可作为 A5 SwiGLU MoE 的基准实现。

该 PR 从首个提交 `00701ca31` 到当前 head `a4853889a` 共包含 20 个提交。后续修正揭示了以下必须保留的工程约束：

| 修正方向 | 设计结论 |
|---|---|
| 量化类型识别 | 以已实例化 `FusedMoE` 的实际 `QuantType` 为准，不能仅解析模型配置字符串 |
| dtype 和 layout | FP4 权重、E8M0 scale、top-k id 必须在调用前做严格校验和规范化 |
| buffer 复用 | symmetric buffer 挂在进程级通信组上，由全部 MoE 层和兼容的主/草稿模型共享 |
| HCCL 初始化 | MegaMoE 使用独立 process group，并在进入算子前主动创建 communicator |
| DP 不均衡 | 所有 rank 按 active rank 的最大 token 数对齐，idle rank 也必须进入同一次 collective |
| ACLGraph | runtime rank 必须采用一致 graph mode，buffer 容量必须覆盖实际 graph bucket |
| padding | 只 pad 到本轮 DP 最大 token 数，不能每轮 pad 到 buffer 总容量 |
| Prefill 容量 | Prefill 节点按本地 scheduler token budget 估算，Decode 按 graph/MC2 execution capacity 估算 |

### 3.2 recipes 中的 K3 算子契约

`cann-recipes-infer/models/kimi_k3/models/modeling_kimi_k3.py` 的 `_moe_mega_w4a8` 已接入 K3 MegaMoE，关键调用参数为：

```python
mega_moe(
    x=routed_states,
    topk_ids=topk_idx.to(torch.int32),
    topk_weights=topk_weight,
    l1_weights=[w13_weight],
    l1_weights_sf=[w13_weight_scale],
    l2_weights=[w2_weight],
    l2_weights_sf=[w2_weight_scale],
    weight1_type=torch_npu.float4_e2m1fn_x2,
    weight2_type=torch_npu.float4_e2m1fn_x2,
    sym_buffer=sym_buffer,
    activation="situglu",
    activation_params={"beta": beta, "linear_beta": linear_beta},
)
```

recipes 初始化 symmetric buffer 时使用 K3 latent 维度，而不是模型全局 hidden：

```text
num_experts              = 896
num_topk                 = K3 配置值
hidden                   = routed_expert_hidden_size = 3584
intermediate_hidden      = 2 * moe_intermediate_size = 6144
dispatch_quant_mode      = 4
dispatch_quant_out_dtype = FP8 E4M3FN
```

### 3.3 0.26 当前基础

0.26 已经完成 K3 模型结构适配：

- `KimiK3MoE` 在完整 hidden 上计算 Gate。
- `routed_input_transform` 完成 `7168 -> 3584` latent projection。
- `FusedMoE` 的 routed experts 工作在 hidden 3584 上。
- `routed_output_transform` 在专家结果完成全局 combine 后执行 RMSNorm 和 `3584 -> 7168` projection。
- `SituActivationConfig` 已携带 `beta` 和 `linear_beta`。
- 共享专家由现有 `FusedMoE` 逻辑独立计算和归并。

因此 K3 模型文件不需要重构。接入点应放在通用 MoE 通信和量化层。

## 4. 目标与非目标

### 4.1 目标

1. 在 A5、EP>1、K3 MXFP4 场景选择 MegaMoE。
2. 精确保留 K3 的 Gate、SiTU、routed scaling、latent transform 和共享专家语义。
3. 支持 eager 和 ACLGraph 下的 Prefill、Decode。
4. 不影响 DSV4 SwiGLU 路径和其他 MoE 通信方式。
5. 条件不满足时确定性回退到 MC2、AllGather 或 AllToAll。
6. symmetric buffer 在进程内只分配一次，并在全部 K3 MoE 层间复用。

### 4.2 首期非目标

- 运行时解析为 INT W4A8 的 ModelSlim 路由专家。
- BF16 MegaMoE。
- 动态 EPLB、冗余专家和 mix placement。
- MoE LoRA。
- K3 MTP/DSpark 草稿模型的 MegaMoE 加速。
- 超过 buffer capacity 后对一个 batch 做 MegaMoE 分块执行。
- 修改 K3 shared expert 的计算实现。

超出首期范围的配置必须在模型启动或后端选择阶段明确回退或报错，不能进入算子后隐式改变语义。

## 5. 总体设计

### 5.1 控制面

沿用 PR #14449 的逻辑映射：A5 MegaMoE 对外仍表现为 `MoECommType.FUSED_MC2`，内部由 A5 专用 backend 执行。这样可以复用 0.26 中“FUSED_MC2 输出已完成路由归并”的 reduction 语义。

建议保留当前 `use_cann_megamoe()` 作为 A3 legacy 判定，不直接扩展其含义。A5 在 `_select_a5_moe_comm_method()` 中单独选择 MegaMoE，避免解除 A3 因缺陷而设置的全局禁用。

选择条件必须同时满足：

```text
device == A5
enable_fused_mc2 == 1
enable_expert_parallel == true
EP world size > 1
CANN ops-transformer 提供 mega_moe/get_symm_buffer_for_mega_moe
resolved QuantType == W4A8MXFP             # K3 首期
MXFP group_size == 32
activation is SituActivationConfig
dynamic_eplb == false
num_redundant_experts == 0
mix_placement == false
LoRA disabled
not draft/MTP                                  # K3 首期
max_tokens_across_dp <= buffer_tokens_per_rank
```

为了保留 DSV4 已验证能力，backend 可以继续支持 PR 中的 `MXFP4`、`MXFP8` 和 SwiGLU，但 K3 发布门禁只覆盖 `W4A8MXFP + SiTU`。两种激活必须走显式分支，不允许把 SiTU 归一化为 SwiGLU。

后端选择使用本轮 `max_tokens_across_dp`，不能使用本 rank 的 token 数。选择结果必须在全部 collective rank 上一致。

### 5.2 数据面

```text
set_ascend_forward_context
    |
    | resolve runtime MoE capability and max tokens across DP
    v
select_moe_comm_method -> FUSED_MC2/A5
    |
    v
PrepareAndFinalizeWithMegaMoE.prepare
    | pad hidden/router to active DP max, save original T
    v
MXFP quant method
    | select_experts, normalize weight/scale orientation
    v
A5MegaMoEBackend.fused_experts
    | validate -> acquire shared buffer -> mega_moe(...)
    v
PrepareAndFinalizeWithMegaMoE.finalize
    | unpad to original T
    v
routed_output_transform -> add shared expert
```

MegaMoE 自己拥有 token dispatch/combine，因此 A5 `FusedMC2CommImpl` 不应实例化可执行的 `TokenDispatcherWithMC2`。使用一个抛出明确异常的 bypass dispatcher，避免未来误调用产生双重 dispatch。

## 6. 详细设计

### 6.1 Runtime capability 解析

PR #14449 同时从 model instance 和按 `VllmConfig` 缓存的量化类型中取值。0.26 应迁移这一思路，但扩充为完整的 A5 MoE capability：

```python
@dataclass(frozen=True)
class A5MegaMoECapability:
    quant_type: QuantType
    activation: str | SituActivationConfig
    group_size: int | None
    is_draft_model: bool
```

`AscendMoERunner` 已得到最终 `self.quant_type` 和 runtime activation，应在初始化时缓存 capability。forward context 优先从当前 `model_instance` 解析并缓存结果，配置字符串只作为不影响正确性的 fallback。

K3 的 MoE 挂在 `layer.block_sparse_moe.experts` 下，不能只检查 `layer.mlp.quant_type`。实现上可以让 `KimiK3MoE` 暴露只读 `quant_type`/`activation` 属性，或让 resolver 识别 `block_sparse_moe.experts`。不建议每个 forward 对整个模型调用 `named_modules()`。

当用户显式开启融合但 CANN 包缺少算子 API 时，应在初始化阶段报出包含依赖版本提示的错误。由于量化、激活或 EPLB 不受支持而未命中时，则记录一次回退原因并使用现有路径。

### 6.2 A5MegaMoEBackend

在 `vllm_ascend/ops/fused_moe/mega_moe.py` 新增独立 backend。A3 当前 `_apply_cann_mega_moe()` 的 INT8/BF16 和 per-expert list 契约与 A5 MXFP stacked tensor 契约不同，不应合并为一个实现。

backend 负责：

1. 检查 quant type、MXFP 参数、EPLB 和冗余专家约束。
2. 接受 tensor、list 或 tuple，并统一为非空 tensor list。
3. 将 uint8 scale 零拷贝 reinterpret 为原生 `torch.float8_e8m0fnu`。
4. 校验权重/scale 的 rank、shape、dtype、stride 和 contiguous 状态。
5. 将 `topk_ids` 转成 contiguous INT32；`log2phy` 映射后再次 contiguous。
6. 按激活类型构造算子参数。
7. 获取进程级 symmetric buffer 并调用 `mega_moe`。

K3 调用分支：

```python
if isinstance(activation, SituActivationConfig):
    activation_name = "situglu"
    activation_params = {
        "beta": activation.beta,
        "linear_beta": activation.linear_beta,
    }
    activation_clamp = None
else:
    activation_name = normalize_swiglu(activation)
    activation_params = None
    activation_clamp = positive_swiglu_limit_or_none
```

`linear_beta=None` 是否由当前 CANN API 接受需要在绑定层确认。如果 API 要求 float，应转换为 recipes/算子定义约定的默认值，而不能自行选择数值。

### 6.3 K3 权重和 scale 布局

设本 rank 专家数为 `E`、latent hidden 为 `H=3584`、intermediate 为 `I=3072`。MegaMoE 需要：

```text
w1       [E, 2I, H/2]       = [E, 6144, 1792], packed FP4 uint8
w2       [E, H, I/2]        = [E, 3584, 1536], packed FP4 uint8
w1_scale [E, 2I, H/64, 2]   = [E, 6144, 56, 2], E8M0
w2_scale [E, H, I/64, 2]    = [E, 3584, 48, 2], E8M0
```

当前 W4A8 MXFP4 普通 GMM 路径在 `process_weights_after_loading()` 中将权重和 scale 转为 GMM 方向。参考 PR #14449，在 MegaMoE 调用前对维度 1/2 再做一次 transpose，可恢复 checkpoint/MegaMoE 方向。由于这是对已转置 view 的逆转置，预期恢复 contiguous storage，不产生每 step 大 tensor copy；backend 必须用实际 stride 做断言。

不建议为 MegaMoE 常驻复制第二份专家权重。K3 有大量专家和层，双份权重会产生不可接受的 HBM 开销。

`intermediate_hidden` 从实际 `w1.shape[1]` 推导为 6144。不能复用 `moe_intermediate_size=3072`，也不能从全局 hidden 7168 推导。

### 6.4 SiTU 数值语义

K3 SiTU 定义为：

```text
gate = beta * tanh(gate / beta) * sigmoid(gate)
up   = linear_beta * tanh(up / linear_beta)    # linear_beta 非空时
out  = gate * up
```

PR #14449 的 `_normalize_activation()` 只接受 SiLU/SwiGLU，其 `activation_clamp` 也只表示 DSV4 SwiGLU clamp。K3 必须新增独立 `situglu` 分支并完整传递 `beta`、`linear_beta`。不得用 `activation_clamp` 模拟 SiTU，也不得因名字包含 GLU 而走 SwiGLU。

普通 fallback 继续使用 0.26 现有 `SituActivationConfig` 和 `situ_mx_quant` 路径，以其输出作为精度对照基线。

### 6.5 通信组和 symmetric buffer

新增 `_MEGA_MOE` group，rank 拓扑与 EP-like/MC2 group 相同，但必须使用独立 group name 和物理 communicator。初始化后通过 backend 的 `get_hccl_comm_name()` 等价路径主动完成 HCCL communicator 建立，避免第一次 symmetric buffer collective 与其他 lazy communicator 创建交叉。

buffer 状态挂在 `_MEGA_MOE` group coordinator 上，实现所有 MoE layer 共享：

```python
@dataclass(frozen=True)
class MegaMoEBufferKey:
    ep_ranks: tuple[int, ...]
    num_experts: int
    tokens_per_rank: int
    top_k: int
    hidden: int
    intermediate_hidden: int
    dispatch_quant_mode: int
    dispatch_quant_out_dtype: torch.dtype
```

第一次创建后 key 不允许变化。后续请求 key 不一致时直接报错，不能销毁并重建，因为其他层、graph 或主/草稿模型可能仍引用旧 buffer。

buffer 应在 profile/warmup 阶段首次创建，进入 ACLGraph capture/replay 后禁止创建。进程销毁时由 `destroy_ascend_model_parallel()` 统一释放 `_MEGA_MOE` group。

### 6.6 Token capacity

沿用已有 `mega_moe_max_tokens`，但明确 A5 语义为全 EP 的配置上限。A5 每 rank 输入容量计算为：

```text
configured_per_rank = mega_moe_max_tokens // EP_size

Prefill producer:
    execution_per_rank = scheduler.max_num_batched_tokens

Decode 或非 producer:
    execution_per_rank = graph/MC2 token capacity

buffer_tokens_per_rank = min(configured_per_rank, execution_per_rank)
```

必须区分两个概念：

- A5 `num_max_tokens_per_rank` 是进入 MegaMoE 的每 rank 输入容量。
- A3 `max_recv_token_num` 是 dispatch 后单 rank 最大接收容量。

两者当前共用配置名，但计算公式不能共用。文档和日志应明确平台语义。

当本轮 active DP 最大 token 数超过 A5 buffer 容量时，整个 group 在 forward context 阶段一致回退，不允许某个 rank 单独回退。首期不实现 recipes 的 chunked MegaMoE。

### 6.7 DP、idle rank 和 ACLGraph

MegaMoE 是对称 collective，以下不变量必须同时成立：

1. 所有 rank 进入相同的 MoE backend。
2. 所有 rank 的输入第一维相同。
3. 所有 rank 使用相同 ACLGraph mode 和 graph bucket。
4. 没有业务 token 的 idle rank 仍执行 native DP dummy forward 并进入 MegaMoE。

`_sync_metadata_across_dp()` 需要同步三项数据：token 数、graph mode、是否有真实工作。最大 token 和 graph mode 只基于 active rank 计算，随后把 dummy rank 的执行 token 数修正为 active rank 最大值。

`should_skip_allreduce_across_dp_group()` 在任意运行形态可能选中 A5 MegaMoE 时必须返回 `False`，确保选择前已有统一的 DP metadata。检查时至少探测：

- 最小非空 batch，用于确认配置是否可能进入 MegaMoE。
- 最大 Decode graph/MC2 shape。
- 最大 Prefill scheduler shape。

`PrepareAndFinalizeWithMegaMoE.prepare()` 只 pad hidden states 和 router logits 到本轮 `max_tokens_across_dp`，并记录本 rank 原始 token 数。量化方法随后在 padded 数据上执行 top-k，保证 top-k ids/weights shape 一致。`finalize()` 仅裁剪回原始长度，不再执行额外 reduce。

不要 pad 到完整 buffer capacity。该做法会显著放大 Decode 计算量，也是 PR #14449 后续性能修正的重点。

### 6.8 Shared expert 和 routed transform

MegaMoE 只输出已全局 combine 的 latent routed result `[T, 3584]`。0.26 现有流程继续执行：

1. `routed_output_transform`: RMSNorm + up projection。
2. shared expert 的 TP reduction。
3. shared result 与 routed result 相加。

不能把 routed RMSNorm/up projection放入 MegaMoE。RMSNorm 是非线性的，必须在所有路由专家贡献完成 combine 后执行。

MegaMoE 是单个融合算子，无法提供普通路径的 `before_gmm2` 和 `before_combine` 中间事件。首期返回 `None`，共享专家继续依赖现有 `before_routed_experts` 事件并发执行。性能验证中需要确认 shared stream 没有引入错误等待或默认流环依赖。

### 6.9 回退和失败策略

| 场景 | 行为 |
|---|---|
| 未开启 `enable_fused_mc2=1` | 使用现有 A5 MC2/AllGather/AllToAll |
| 非 MXFP4 K3 checkpoint | 回退并记录一次 quant type 原因 |
| SiTU 参数合法但 CANN API 不支持 `activation_params` | 启动失败，禁止退化成 SwiGLU |
| 动态 EPLB、冗余专家、mix placement、LoRA | 启动期或选择期回退 |
| batch 超过 buffer capacity | 全 rank 一致回退 |
| buffer key 在运行期变化 | RuntimeError，禁止重建 |
| 权重 shape/dtype/stride 不匹配 | 第一次执行前 RuntimeError |
| 缺失 CANN MegaMoE API | 显式开启时启动失败并提示依赖版本 |

日志策略：初始化、命中和永久回退原因使用 `info_once`；每 step 的 shape、选择和 buffer 复用信息只使用 `DEBUG`，避免复现 PR review 中指出的热路径日志开销。

## 7. 文件级改动方案

| 文件 | 计划改动 |
|---|---|
| `vllm_ascend/ascend_forward_context.py` | 增加 A5 capability 解析、capacity 计算和 FUSED_MC2 选择；将 `model_instance` 传入 selector |
| `vllm_ascend/ascend_config.py` | 增加 A5 per-rank capacity 纯函数，补充 `mega_moe_max_tokens` 平台语义校验 |
| `vllm_ascend/distributed/parallel_state.py` | 创建/销毁独立 MegaMoE group，主动初始化 HCCL communicator |
| `vllm_ascend/ops/fused_moe/mega_moe.py` | 新增 A5 backend、参数规范化、layout 校验、SiTU 参数和共享 buffer |
| `vllm_ascend/ops/fused_moe/prepare_finalize.py` | 新增 `PrepareAndFinalizeWithMegaMoE` |
| `vllm_ascend/ops/fused_moe/moe_comm_method.py` | A5 FUSED_MC2 路由到 backend；A3 逻辑保持隔离 |
| `vllm_ascend/ops/fused_moe/fused_moe.py` | 缓存实际 capability；保留 routed transform/reduction 语义 |
| `vllm_ascend/quantization/methods/w4a8_mxfp4.py` | FUSED_MC2 下恢复 MegaMoE 权重和 scale 方向并传入 runtime args |
| `vllm_ascend/worker/model_runner_v1.py` | 同步 active/dummy DP metadata 和 graph mode |
| `vllm_ascend/utils.py` | MegaMoE 可选时禁止跳过 DP metadata all-reduce |
| `tests/ut/ops/test_mega_moe.py` | backend、SiTU、layout、buffer 和 prepare/finalize 单测 |
| `tests/ut/test_ascend_forward_context.py` | A5 选择/回退/容量/跨 rank 一致性单测 |
| `tests/ut/quantization/methods/test_w4a8_mxfp4.py` | K3 MXFP4 权重方向、scale dtype 和参数透传单测 |
| `tests/ut/worker/a2/test_model_runner_v1.py` | DP active/dummy rank 与 graph mode 同步单测，目录名沿用现有测试组织 |

不建议直接修改 `vllm_ascend/models/kimi_k3.py` 的计算图。如果 capability resolver 需要稳定入口，只增加只读元数据属性，不改变 forward。

## 8. 测试与验证

### 8.1 单元测试

1. A5 + EP + `W4A8MXFP` + SiTU + group size 32 命中 FUSED_MC2。
2. A3 仍保持当前禁用状态，A2/310P 行为不变。
3. LoRA、dynamic EPLB、冗余专家、mix placement、draft model 和超容量场景正确回退。
4. K3 operator 参数包含 `activation="situglu"` 和完整 `activation_params`。
5. K3 buffer key 使用 hidden 3584、intermediate hidden 6144，而非 7168/3072。
6. FP4 packed weight、4D E8M0 scale、INT32 contiguous top-k ids 校验。
7. 普通 GMM layout 经逆 transpose 后与 MegaMoE 期望 shape/stride 一致。
8. 92 层重复创建 backend 时只分配一份 symmetric buffer。
9. buffer key 不一致和输入超过容量时 fail fast。
10. 不同 DP token 数、idle rank 和不同本地 graph mode 最终得到统一执行 shape/mode。

### 8.2 A5 算子级验证

固定同一组 latent input、router logits 和专家权重，对比：

- 0.26 普通 W4A8 MXFP4 FusedMoE + SiTU fallback。
- recipes `_moe_mega_w4a8`。
- vLLM Ascend A5 MegaMoE backend。

比较 routed latent output，再单独比较 RMSNorm/up projection 后输出。这样可以区分算子内部误差和模型外层误差。

### 8.3 K3 32 卡模型验证

首期标准拓扑为 4 节点 32 卡、EP=32，至少覆盖：

| 维度 | 用例 |
|---|---|
| 执行阶段 | Decode-only、短 Prefill、最大 scheduler Prefill、chunked Prefill 混合 batch |
| 执行模式 | eager、ACLGraph Decode、ACLGraph capture/replay |
| DP 负载 | rank 等长、明显不等长、部分 idle rank |
| 输入长度 | 短输入、长输入、接近 buffer capacity、超过 capacity 回退 |
| shared expert | 单流、现有 shared expert multistream 配置 |
| 精度 | 逐层 routed output 抽检、最终 logits、固定 prompt greedy generation |

精度基准使用关闭 MegaMoE 的同一 0.26 commit 和同一 checkpoint。验收指标应包含：

- 无 NaN/Inf。
- top-k token 一致率和 logits 误差达到项目 MXFP4 现有门限。
- 固定 greedy case 输出一致或满足已定义量化误差标准。
- 连续压力运行无 HCCL timeout、507057、graph replay shape 错误或 buffer 重建。

### 8.4 回归与性能

- 复跑 PR #14449 已验证的 DSV4 A5 SwiGLU 场景，防止新增 SiTU 分支破坏原能力。
- 验证未开启融合时 K3 和其他 MoE 模型行为不变。
- 采集 Prefill throughput、Decode latency、单步 MegaMoE 时间、HCCL 时间、HBM 占用和 graph capture 开销。
- 确认热路径没有 INFO 级逐 step shape 日志。

## 9. 分阶段交付

### Phase 1: 后端和算子契约

- 新增独立 A5 backend。
- 完成 K3 SiTU、FP4/E8M0 layout 和算子参数单测。
- 使用 mock operator 验证调用契约，不启用运行时选择。

### Phase 2: 通信、buffer 和 DP

- 新增独立 MegaMoE group 和进程级 buffer。
- 增加 prepare/finalize、DP active/dummy metadata 同步。
- 完成 eager 多卡算子验证。

### Phase 3: K3 选择和模型验证

- 打开 A5 `W4A8MXFP + SiTU` capability gate。
- 完成 32 卡 Prefill/Decode 精度和稳定性验证。
- 验证超容量和不支持配置的确定性回退。

### Phase 4: ACLGraph 和灰度

- 完成 graph capture/replay 和不均衡 DP 验证。
- 复跑 DSV4 回归与性能对比。
- 保持 `enable_fused_mc2=1` opt-in，达到稳定性门限后再讨论默认开启。

每个 Phase 独立提交，避免把 0.23 PR 的 29 文件大 diff 一次性迁入 0.26。

## 10. 风险与缓解

| 风险 | 等级 | 缓解措施 |
|---|---|---|
| 不同 rank 选择不同 backend 或 shape | 最高 | 选择前同步 DP metadata；只使用 active rank 最大值；统一 dummy 执行 |
| CANN MegaMoE API 与 recipes 版本漂移 | 最高 | 固定并记录 ops-transformer/CANN 版本；启动期校验 `activation_params` 能力 |
| SiTU 被错误映射为 SwiGLU | 高 | 类型分支和独立 golden test；不允许语义降级 |
| MXFP4 packed layout/scale stride 不匹配 | 高 | 第一次调用严格校验实际 shape、dtype、stride、contiguous |
| buffer 容量不足或过大 | 高 | 区分 P/D 容量来源；超限全组回退；记录一次容量来源 |
| graph capture 时创建通信资源 | 高 | warmup 前完成 group 初始化；capture/replay 禁止 buffer 创建 |
| shared expert stream 与融合算子互锁 | 中 | 不伪造中间 event；覆盖单流/多流压力测试 |
| 全局 capability cache 污染主/草稿模型 | 中 | cache 绑定 config/model identity；首期 K3 draft 不启用 MegaMoE |

## 11. 备选方案与取舍

### 11.1 整体替换 KimiK3MoE

不采用。0.26 的 `FusedMoE` 已正确表达 latent input/output transform、共享专家和 reduction 语义。复制 recipes 的整层实现会绕过 vLLM 的量化、graph、DP 和 fallback 框架，维护成本高。

### 11.2 直接 cherry-pick PR #14449

不采用。该 PR 基于 0.23，包含 `fused_moe_0_23_0.py` 等兼容代码；0.26 的 MoE runtime args、K3 SiTU 和模型结构均已变化。应按模块迁移设计和测试，而不是搬运 diff。

### 11.3 将 A5 逻辑加入现有 A3 MegaMoE 函数

不采用。A3 当前使用 INT8/BF16、FRACTAL_NZ 和 per-expert list，A5 使用 stacked MXFP tensor、E8M0 scale 和不同 buffer 语义。共用实现会增加平台条件分支并扩大 A3 回归范围。

### 11.4 超容量时自动 chunk MegaMoE

首期不采用。recipes 已实现 collective 顺序一致的 chunk plan，但 vLLM 在线调度还需要处理 graph bucket、DP idle rank 和回退一致性。首期整 batch 回退更容易证明正确，后续再作为独立性能功能实现。

## 12. 待确认项

1. 用于发布验证的 K3 checkpoint 是否在运行时解析为 `W4A8_MXFP` group size 32。若 ModelSlim checkpoint 最终解析为 INT W4A8，本设计不能直接启用。
2. 目标 CANN/ops-transformer 版本及 `mega_moe` Python 签名，特别是 `situglu.activation_params` 和 `linear_beta=None` 的约定。
3. 32 卡拓扑是否固定 EP=32，以及 P/D 分离时 Prefill 和 Decode 的 scheduler/graph token 上限。
4. 项目对 K3 MXFP4 logits、token 一致率和性能收益的正式验收阈值。
5. K3 MTP/DSpark 是否纳入后续版本；若纳入，需要先证明主/草稿模型可以共享相同 buffer key，或确保草稿模型不进入该 group collective。

## 13. 验收标准

只有同时满足以下条件才允许从实验状态进入默认支持矩阵：

1. K3 32 卡 eager 和 ACLGraph 的 Prefill/Decode 精度通过。
2. DP 不均衡和 idle rank 压力测试无 collective hang。
3. 最大支持 batch 不发生 buffer overflow，超限场景全组一致回退。
4. 进程内 symmetric buffer 只创建一次，全部 MoE 层复用。
5. DSV4 A5 回归通过，A2/A3/310P 行为不变。
6. 关闭功能开关时，与 0.26 基线在行为和性能上无显著回退。
7. CANN/ops-transformer 依赖版本、启动参数和已知限制有明确文档。

## 14. 实现与验证状态

当前实现已完成以下代码路径：

- A5 capability 使用实际 `QuantType`、运行时激活和 MXFP group size 判定；首期仅允许
  `W4A8MXFP + group_size=32`，K3 draft/MTP、LoRA、dynamic EPLB、冗余专家和 mix
  placement 均回退。
- A5 `FUSED_MC2` 使用独立 MegaMoE group、bypass dispatcher 和进程级不可变
  symmetric buffer；buffer 在 warmup 创建，ACLGraph capture 内禁止首次创建。
- K3 `SituActivationConfig` 映射为 `activation="situglu"`，并原样传递 `beta` 和
  `linear_beta`；不使用 SwiGLU clamp 模拟 SiTU。
- W4A8 MXFP4 权重和 scale 在调用点恢复 checkpoint 方向，不保留第二份常驻专家权重。
- DP metadata 同步 token 数、graph mode 和 active/dummy 状态；MegaMoE prepare 只 pad
  到本轮 active DP 最大 token 数，finalize 只裁剪本 rank 原始长度。

本地环境仅有源码，不包含 `torch`、`vllm`、`torch_npu`、`pytest`、NPU 或 K3 checkpoint。
已完成 Ruff、codespell、typos、Markdown lint、Python 语法编译和仓库本地静态检查；未执行
算子、服务或模型精度测试。gitleaks 因本地无二进制且网络 DNS 受限未执行。

| 特性 | 当前状态 | 说明 |
|---|---|---|
| EP=32 | 已实现，待硬件验证 | 独立 MegaMoE group 与全 rank 一致选择已接入 |
| ACLGraph | 已实现，待硬件验证 | warmup buffer、capture 禁止创建和 active graph mode 同步已接入 |
| flashcomm1 | 未验证 | 不属于本次算子替换门禁，首轮验证建议关闭以隔离变量 |
| MTP/DSpark | 首期不支持 | draft path 明确回退，不进入 MegaMoE collective |
| 多模态 | 模型路径未改，待验证 | 至少需要一条文本请求和一条图文请求验证 |
| 128K + bs16 | 未验证 | 缺少 4 节点 32 卡 A5 环境和真实权重 |

| 验证阶段 | 本地结果 | A5 发布门禁 |
|---|---|---|
| dummy 权重 | 未执行 | 必须完成启动、`/v1/models`、文本和图文请求 |
| 真实权重 | 未执行 | 必须完成非空输出、逐层 routed output、logits 和稳定性验证 |

dummy 只能验证架构、算子调用和 API 路径，不能验证权重 key、MXFP4 layout、scale dtype
或最终精度。没有真实权重结果时不得把本实现标记为发布验收通过。本地也没有发生
false-ready；原因是服务未启动，而不是已排除首次请求失败。

## 15. A5 验证运行手册

以下命令是 4 节点 32 卡、TP=8、DP=4、EP=32 的验证模板。所有节点使用同一代码、模型、
CANN 和 ops-transformer 版本，并确认 `cann_ops_transformer.ops` 同时导出 `mega_moe` 和
`get_symm_buffer_for_mega_moe`。在 `/workspace` 运行，首轮关闭 flashcomm1 和 MTP。

Node 0：

```bash
cd /workspace
export MODEL_PATH=<KIMI_K3_MXFP4_MODEL_PATH>
export LOCAL_IP=<NODE0_IP>
export NIC_NAME=<NODE0_NIC_NAME>
export RPC_PORT=<DP_RPC_PORT>
export HCCL_IF_IP=$LOCAL_IP
export HCCL_SOCKET_IFNAME=$NIC_NAME
export GLOO_SOCKET_IFNAME=$NIC_NAME
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0

vllm serve "$MODEL_PATH" \
    --served-model-name kimi-k3 \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --data-parallel-size 4 \
    --data-parallel-size-local 1 \
    --data-parallel-address "$LOCAL_IP" \
    --data-parallel-rpc-port "$RPC_PORT" \
    --enable-expert-parallel \
    --max-model-len 131072 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 2048 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_fused_mc2":1,"mega_moe_max_tokens":65536}' \
    2>&1 | tee /workspace/kimi-k3-a5-megamoe.log
```

Nodes 1-3 使用相同参数，增加 `--headless`，把 `--data-parallel-address` 设置为 Node 0 IP，
并分别设置 `--data-parallel-start-rank 1`、`2`、`3`。dummy gate 增加
`--load-format dummy`；真实权重 gate 必须删除该参数并重新启动。

服务就绪与文本 smoke：

```bash
curl -f http://127.0.0.1:8000/v1/models
curl -f http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"kimi-k3","messages":[{"role":"user","content":"say hi"}],"temperature":0,"max_tokens":16}'
```

图文 smoke 使用 `Kimi-K3.md` 的 OpenAI 兼容图像请求，并要求 HTTP 200、非空 `choices`。
日志至少检查以下证据：

```bash
rg -n "A5 MegaMoE runtime capability|A5 MegaMoE backend initialized|symmetric buffer|Replaying aclgraph" \
    /workspace/kimi-k3-a5-megamoe.log
```

若 graph 路径失败，保持其他参数不变并增加 `--enforce-eager` 做隔离；不得把 eager 成功等同于
ACLGraph 成功。算子或首次请求失败时保留完整日志、固定 prompt 和同一 checkpoint，再与关闭
`enable_fused_mc2` 的普通 W4A8 MXFP4 路径对比。
