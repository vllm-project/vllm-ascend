# Kimi K3 A5 MegaMoE 算子替换说明

## 1. 文档目的

本文说明 vLLM Ascend 0.26 中 Kimi K3 路由专家的 A5 MegaMoE 替换，包括：

- 替换范围和保留范围；
- SiTU-GLU 激活的参数与数值语义；
- 后端选择条件和回退行为；
- MXFP4 权重、scale 和 symmetric buffer 契约；
- DP、idle rank 和 ACLGraph 的一致性要求；
- 如何确认运行时实际进入 MegaMoE；
- 当前已实现能力与尚未完成的硬件验收。

本文描述的是当前实现，不代表功能已经通过发布验收。完整方案、历史取舍和风险分析见
[Kimi K3 A5 MegaMoE 适配设计](kimi_k3_a5_megamoe.md)。

## 2. 支持范围

当前实现是 Kimi K3 A5 MegaMoE 的首期受限适配，不是所有 Kimi K3 checkpoint 和运行模式的
通用替换。

| 项目 | 当前范围 |
|---|---|
| 硬件 | Ascend A5，即 Atlas 950PR/950DT |
| 模型路径 | Kimi K3 主模型 routed experts |
| 量化类型 | 运行时解析为 `QuantType.W4A8MXFP` |
| 权重量化 | MXFP4 E2M1 packed weight |
| 激活量化 | 动态 MXFP8 E4M3FN |
| MXFP group size | `32` |
| 激活 | `SituActivationConfig`，映射为 `situglu` |
| 并行 | Expert Parallel，EP world size 大于 1 |
| 执行模式 | eager 和 ACLGraph 代码路径已接入，待 A5 实机验证 |
| 开关 | `additional_config.enable_fused_mc2=1` |

以下场景不会使用 A5 MegaMoE：

- routed experts 解析为普通 INT `W4A8`，而不是 `W4A8MXFP`；
- MTP 或 DSpark draft model；
- dynamic EPLB、冗余专家或 mix placement；
- MoE LoRA；
- EP world size 等于 1；
- 当前 batch 超过 A5 symmetric buffer 的 rank-local token capacity。

这些限制用于确保所有 collective rank 作出相同的后端选择。条件不满足时，运行时回退到现有
MC2、AllGather 或 AllToAll 路径，不把不支持的配置送入 MegaMoE。

如果用户显式开启 `enable_fused_mc2=1`，但安装的 `cann_ops_transformer.ops` 不提供
`mega_moe` 或 `get_symm_buffer_for_mega_moe`，模型初始化会直接失败并提示升级依赖。依赖缺失
不属于可静默回退的运行时能力差异。

## 3. 替换边界

Kimi K3 使用 Stable LatentMoE。Gate 和 shared expert 工作在完整 hidden size 7168，routed
experts 工作在 latent hidden size 3584。MegaMoE 只替换 routed experts 内部的通信和专家
计算，不替换整层 Kimi K3 MoE。

```text
full hidden [T, 7168]
    |
    |-- Gate ---------------------------------------> top-k ids / weights
    |-- routed_expert_down_proj -------------------> [T, 3584]
    |                                                   |
    |                                                   v
    |                          +------------------------------------------+
    |                          | A5 MegaMoE                              |
    |                          | Dispatch -> GMM1 -> SiTU -> GMM2        |
    |                          |          -> weighted Combine             |
    |                          +------------------------------------------+
    |                                                   |
    |                                                   v
    |                                               [T, 3584]
    |                                                   |
    |                          routed RMSNorm -> routed_expert_up_proj
    |                                                   |
    |                                                   v
    |                                               [T, 7168]
    |
    |-- shared expert [T, 7168] ------------------------+
                                                        |
                                                        v
                                                  final MoE output
```

MegaMoE 输出已经完成跨 EP rank 的 routed expert combine。随后必须先执行 routed RMSNorm，
再执行 latent up projection。RMSNorm 是非线性运算，不能移动到 combine 之前。Shared expert
继续使用原有实现，在完整 hidden size 上计算并与 routed output 相加。

## 4. SiTU-GLU 语义

Kimi K3 的 routed experts 不是普通 SwiGLU，而是 SiTU-GLU。设 GMM1 输出的 gate 分支为
`gate`，linear/up 分支为 `up`：

```text
gate_out = beta * tanh(gate / beta) * sigmoid(gate)

linear_out = up                                      # linear_beta is None
linear_out = linear_beta * tanh(up / linear_beta)   # otherwise

output = gate_out * linear_out
```

模型构造阶段使用 `SituActivationConfig` 保存参数：

```python
SituActivationConfig(
    beta=config.activation_situ_beta or 1.0,
    linear_beta=config.activation_situ_linear_beta,
)
```

A5 backend 对该类型使用独立分支：

```python
activation = "situglu"
activation_params = {
    "beta": activation.beta,
    "linear_beta": activation.linear_beta,
}
activation_clamp = None
```

[`cann_ops_transformer` MegaMoE](https://gitcode.com/cann/ops-transformer/tree/master/mc2/mega_moe)
的 Torch wrapper 接受上述命名参数字典，并在进入底层算子前转换为 `[beta]` 或
`[beta, linear_beta]`。实现不能进行以下语义替换：

- 不能把 `SituActivationConfig` 归一化成 `swiglu`；
- 不能使用 `activation_clamp` 模拟 `beta`；
- 不能丢弃 `linear_beta`；
- 如果目标 ops-transformer 不接受 `activation_params`，必须报错，不能静默退化。

## 5. 后端选择

`set_ascend_forward_context()` 在模型执行前使用所有 active DP rank 的最大 token 数调用
`select_moe_comm_method()`。A5 只有同时满足以下条件才返回逻辑
`MoECommType.FUSED_MC2`：

```text
device == A5
enable_fused_mc2 == 1
enable_expert_parallel == true
EP world size > 1
resolved quant type == W4A8MXFP
MXFP group size == 32
activation is supported, including SituActivationConfig
dynamic EPLB == false
num redundant experts == 0
mix placement == false
LoRA disabled
not draft/MTP
max tokens across active DP ranks <= buffer tokens per rank
```

量化类型、激活和 group size 来自已实例化的 `AscendMoERunner`，并缓存在对应
`VllmConfig` 上。选择逻辑不根据模型名称或 checkpoint 文件名猜测量化类型。因此，即使模型
名包含 `w4a8`，只有运行时实际解析为 `W4A8MXFP` 才能进入该路径。

## 6. 执行流程

一次 routed expert forward 的执行顺序如下：

1. `set_ascend_forward_context()` 同步 DP metadata 并选择通信后端。
2. `PrepareAndFinalizeWithMegaMoE.prepare()` 记录本 rank 原始 token 数。
3. Hidden states 和 router logits 只 pad 到本轮 active DP 最大 token 数。
4. 量化方法在 pad 后的数据上执行 top-k，保证 collective rank 的 tensor shape 一致。
5. W4A8 MXFP 方法恢复 MegaMoE 需要的 checkpoint 权重与 scale 方向。
6. `MegaMoEBackend` 校验 quant type、group size、dtype、shape、stride 和 top-k 契约。
7. Backend 获取进程级 symmetric buffer，并调用 `mega_moe()`。
8. `PrepareAndFinalizeWithMegaMoE.finalize()` 裁剪到本 rank 原始 token 数。
9. Kimi K3 执行 routed RMSNorm 和 latent up projection。
10. 原有 FusedMoE 流程归并 shared expert output。

MegaMoE 自己包含 Dispatch 和 Combine，因此 A5 `FUSED_MC2` 使用 bypass dispatcher。任何对
bypass dispatcher 的调用都会抛出异常，用于防止未来重构意外执行两次 Dispatch 或 Combine。

## 7. 权重与 Scale 布局

设本 rank local expert 数为 `E`，Kimi K3 routed latent hidden 为 `H=3584`，MoE
intermediate size 为 `I=3072`。MegaMoE 输入布局为：

| Tensor | Shape | Storage/semantic dtype |
|---|---|---|
| `w1` | `[E, 2I, H/2] = [E, 6144, 1792]` | packed FP4，一字节 storage |
| `w2` | `[E, H, I/2] = [E, 3584, 1536]` | packed FP4，一字节 storage |
| `w1_scale` | `[E, 2I, H/64, 2] = [E, 6144, 56, 2]` | E8M0 |
| `w2_scale` | `[E, H, I/64, 2] = [E, 3584, 48, 2]` | E8M0 |

普通 grouped-matmul 路径在加载后保存 `(E, K, N)` view。MegaMoE 调用点对维度 1 和 2
执行逆 transpose，恢复 `(E, N, K)` checkpoint 方向。由于这是可逆 view，不为 92 个 MoE
层常驻复制第二份专家权重。

Scale 如果以 `torch.uint8` storage 加载，会零拷贝 reinterpret 为
`torch.float8_e8m0fnu`。Backend 要求逆 transpose 后的权重与 scale contiguous；不满足时在
第一次算子调用前报错。

## 8. 通信组与 Buffer 生命周期

A5 MegaMoE 使用独立 `_MEGA_MOE` group。它与 EP-like/MC2 group 使用相同 rank 拓扑，但
具有独立 group name 和物理 communicator。模型并行初始化时主动建立 HCCL communicator，
避免首次 symmetric buffer collective 与其他 lazy communicator 初始化交叉。

Symmetric buffer 状态保存在 `_MEGA_MOE` group coordinator 上，进程内所有兼容 MoE 层
共享一份。Buffer key 包含：

- EP ranks；
- 全局 expert 数；
- rank-local token capacity；
- top-k；
- latent hidden 和 GMM1 输出宽度；
- dispatch quant mode 和输出 dtype。

首次创建后 key 不可变化。后续请求 key 不一致时直接报错，不在运行期销毁并重建。ACLGraph
capture 内禁止首次创建 buffer；profile/warmup 必须提前完成初始化。

## 9. Token Capacity 与 DP 对齐

`mega_moe_max_tokens` 在 A5 上表示全 EP 的输入 token 配置上限：

```text
configured_per_rank = mega_moe_max_tokens // EP_size

Prefill producer:
    execution_per_rank = scheduler.max_num_batched_tokens

Decode or mixed role:
    execution_per_rank = graph/MC2 token capacity

buffer_tokens_per_rank = min(configured_per_rank, execution_per_rank)
```

超出 capacity 的 batch 在进入算子前统一回退；首期不实现 MegaMoE chunking。

MegaMoE 是全组 collective。DP metadata 同步 token 数、ACLGraph mode 和 rank 是否有真实
业务。最大 token 数和 graph mode 只从 active rank 计算，idle rank 随后使用相同执行 shape
进入 native dummy forward。这样可以避免不同 rank 选择不同后端、不同 graph bucket 或不同
collective 顺序。

## 10. 开启方式

以下仅展示与 MegaMoE 相关的关键参数。完整多节点配置仍应参考 Kimi K3 模型部署文档。

```bash
vllm serve <KIMI_K3_MXFP4_MODEL_PATH> \
    --served-model-name kimi-k3 \
    --tensor-parallel-size 8 \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --max-model-len 131072 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 2048 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_fused_mc2":1,"mega_moe_max_tokens":65536}'
```

首期验证不要启用 MTP/DSpark，也建议先关闭 flashcomm1 以减少变量。对于 EP=32，默认
`mega_moe_max_tokens=65536` 对应 `configured_per_rank=2048`。更大的 batch 会回退到普通路径，
而不是继续使用一个容量不足的 symmetric buffer。

## 11. 如何确认替换生效

使用 `VLLM_LOGGING_LEVEL=DEBUG` 启动，并检查以下日志：

```bash
rg -n \
  "A5 MegaMoE runtime capability|A5 MegaMoE selection|A5 MegaMoE backend initialized|symmetric buffer" \
  <SERVER_LOG>
```

判定要求：

1. Capability 日志显示 `quant_type=QuantType.W4A8MXFP`、SiTU activation 和
   `group_size=32`。
2. Selection 日志显示 `enabled=True` 和 `method=MoECommType.FUSED_MC2`。
3. Backend initialization 日志出现。
4. 第一次 warmup 出现 symmetric buffer 创建日志；后续层只复用该 buffer。
5. ACLGraph 模式下存在 capture/replay 证据，且 capture 内没有首次 buffer 创建。

仅看到服务启动成功，或者只看到 `FUSED_MC2`，都不足以证明 Kimi K3 已执行 MegaMoE。

## 12. 验证要求与当前状态

功能验收必须对比同一 commit、同一真实 checkpoint 下关闭 MegaMoE 的普通 W4A8 MXFP4
路径。至少覆盖 routed latent output、routed transform 后输出、最终 logits 和固定 greedy
生成结果。

| 验证项 | 当前状态 |
|---|---|
| SiTU 参数与算子调用 mock test | 已覆盖 |
| K3 3584/3072 权重与 scale shape test | 已覆盖 |
| Buffer 单次创建和 key 不变 test | 已覆盖 |
| DP active/dummy metadata test | 已覆盖 |
| Python compile、Ruff 和文档检查 | 已通过 |
| A5 EP=32 eager | 未执行 |
| A5 ACLGraph capture/replay | 未执行 |
| Kimi K3 dummy 服务请求 | 未执行 |
| Kimi K3 真实权重文本与图文请求 | 未执行 |
| 128K context、`max-num-seqs=16` | 未执行 |
| 普通路径与 MegaMoE 精度对比 | 未执行 |
| 连续压力与 HCCL 稳定性 | 未执行 |

Dummy 权重只能验证架构、算子和 API 调用路径，不能验证 MXFP4 weight key、实际 storage
layout、scale dtype 或模型精度。没有真实权重 HTTP 200、非空输出和精度结果时，不能宣称
Kimi K3 MegaMoE 已完整支持。

## 13. 相关代码

| 文件 | 职责 |
|---|---|
| `vllm_ascend/ascend_forward_context.py` | Capability 缓存、A5 后端选择和 token capacity |
| `vllm_ascend/ascend_config.py` | A5 rank-local capacity 计算和配置校验 |
| `vllm_ascend/distributed/parallel_state.py` | 独立 MegaMoE group 和 communicator 生命周期 |
| `vllm_ascend/ops/fused_moe/mega_moe.py` | SiTU 参数、layout 校验、buffer 复用和算子调用 |
| `vllm_ascend/ops/fused_moe/prepare_finalize.py` | DP shape padding 和输出裁剪 |
| `vllm_ascend/ops/fused_moe/moe_comm_method.py` | A5 `FUSED_MC2` backend 和 bypass dispatcher |
| `vllm_ascend/ops/fused_moe/fused_moe.py` | Runtime capability 缓存和 routed/shared 语义 |
| `vllm_ascend/quantization/methods/w4a8_mxfp4.py` | 权重方向恢复和 MXFP 参数透传 |
| `vllm_ascend/worker/model_runner_v1.py` | Active/dummy DP metadata 和 warmup shape |
| `vllm_ascend/utils.py` | MegaMoE 可选时保留 DP metadata 同步 |
