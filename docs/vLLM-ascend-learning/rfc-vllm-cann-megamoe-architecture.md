# rfc/vllm_cann 原始分支：MegaMoE 接入架构与执行分析

分析基线：`rfc/vllm_cann`，提交 `7e37e29a7803d6897a7113088e2b771f83205bc0`。分析日期：2026-10-23。

本文只分析这个原始分支，不把 `integrate/dsnext-megamoe-vllm-cann` 的四个适配提交视为已有功能。所有本仓代码路径从 `vllm_ascend/` 开始；链接指向仓库文件，表中的行号对应本次基线。

证据范围是 Python/C++ 接入代码、配置和现有测试的静态阅读。没有在本机执行 NPU 算子；外部 `cann_ops_transformer` 包没有作为本仓源码的一部分提供。因此“接入路径存在”“配置允许”“测试覆盖某场景”和“某个已安装算子版本能正确执行”需要分别判断。

## 1. 先给出整体结论

MegaMoE 接入位于 MoE 层的路由专家执行部分。它替代普通路径里的 token dispatch、专家 MLP 和 token combine，路由打分和 Top-K 仍由外层执行，输出整理和共享专家合并也仍有外层工作。

| 问题 | 当前原始分支的结论 |
| --- | --- |
| A3 是否有 MegaMoE 入口 | 有。硬件声明支持，策略为 `FUSED_OR_CAPACITY`；量化主要看 INT W8A8/W4A8 |
| A5 是否有 MegaMoE 入口 | 有。硬件声明支持 MXFP MegaMoE，策略为 `CAPACITY_AND_WORLD_SIZE` |
| A2 是否会通过当前标准策略选 MegaMoE | 不会。标准硬件 profile 没有 `CANN_MEGAMOE`，选择器只返回 MC2/ALLGATHER；工具函数有 A2/A3 字样不等于 A2 已打通 |
| 310P 是否选择 MegaMoE | 不会。固定 ALLGATHER，并有专用组件 |
| 共享专家能否运行 | 能。常规实现为 `AscendSharedExperts` 独立 MLP 路径 |
| 共享专家是否作为单独权重融合进 MegaMoE 调用 | 当前标准路径没有该接口，调用中没有 shared weight 参数；不能套用 PR !51 的能力 |
| `mix_placement` 是否也是共享专家相关能力 | 是另一套机制：把共享专家作为专家池中的物理专家参与路由执行；不能直接等同于共享权重 ABI 融合，见第 5 章 |
| `FUSED_MC2` 是否必然等于 MegaMoE | 否。它还可能执行本仓 `dispatch_ffn_combine` |
| 启动配置是多少 | 外部 `enable_fused_mc2=2` 尝试 MegaMoE；外部 1 主动关闭 MegaMoE 标志；2 在内部被归一化为 1 |
| A5 `4608` 是否允许 | 默认不允许。`intermediate_hidden=2*moe_intermediate_size` 的白名单不含 4608 |
| 原始 A5 选择器是否按当前 token 容量回退 | 融合分支没有这个检查；选中后在调用入口检查每 rank 输入容量，超限抛异常 |
| MegaMoE 对称 buffer 是否就是 KV Cache | 不是。它是路由通信/计算资源，和 KV Cache、权重、HCCL 常规通信 buffer 分属不同预算 |

最需要保留的四层区分是：

```text
配置申请启用 MegaMoE
        ↓
配置校验与硬件/并行能力允许
        ↓
本次 forward 选择 FUSED_MC2，且 use_mega_moe 为真
        ↓
实际进入 cann_ops_transformer.ops.mega_moe(...)
```

任意一层满足，都不自动证明下一层已经成功；最后一层发生在图捕获期间，也不自动证明每种正式请求形状都走它。

## 2. 架构分层与各仓库的边界

```text
CPU：Worker 进程中的模型执行线程
│
├─ Model Runner V1 / V2                         [vllm + vllm-ascend]
│    └─ Model.forward                          [模型实现]
│         └─ 某层 MoE
│              └─ AscendMoERunner               [vllm-ascend]
│                   ├─ Gate → Router / Top-K
│                   ├─ AscendRoutedExperts
│                   │    ├─ MoECommMethod.prepare
│                   │    ├─ quant_method.apply
│                   │    │    └─ FusedMC2CommImpl.fused_experts
│                   │    │         ├─ get_fused_mc2_weights
│                   │    │         └─ _apply_cann_mega_moe
│                   │    │              └─ mega_moe(...)
│                   │    └─ MoECommMethod.finalize
│                   ├─ AscendSharedExperts（如果存在）
│                   └─ 合并输出、必要的归约和变换
│
└─ Python 调用提交设备任务
         ↓
NPU：dispatch + 专家矩阵计算/激活/量化 + combine  [外部算子包/CANN/HCCL]
```

这些类不是独立线程。通常由同一 Worker 的 CPU 执行线程逐层调用；NPU 工作异步提交到 stream。辅助线程和辅助 stream 的存在，不改变这里的对象调用关系。

| 层次 | 负责内容 | 关键入口 |
| --- | --- | --- |
| 上游 vLLM | 配置总对象、MoE 工厂、基础 Runner/Router、分布式和图执行框架 | 上游 `vllm/model_executor/layers/fused_moe/` |
| Ascend 平台/注册 | 替换 MoERunner、RoutedExperts，补充 forward context 和硬件策略 | [vllm_ascend/utils.py](../../vllm_ascend/utils.py)，`register_ascend_customop()`，906 行附近 |
| Ascend MoE 编排 | Router、共享专家、路由专家之间的顺序和输出语义 | [vllm_ascend/ops/fused_moe/fused_moe.py](../../vllm_ascend/ops/fused_moe/fused_moe.py) |
| Ascend 量化层 | 创建/加载权重、布局变换、构造统一权重载荷 | [vllm_ascend/quantization/method_adapters.py](../../vllm_ascend/quantization/method_adapters.py) 及 `methods/` |
| Ascend 融合通信层 | buffer 参数、HCCL 组、MegaMoE ABI、输出结果封装 | [vllm_ascend/ops/fused_moe/moe_comm_method.py](../../vllm_ascend/ops/fused_moe/moe_comm_method.py) |
| 外部算子包 | `get_symm_buffer_for_mega_moe`、`mega_moe`、通信上下文扩展 | 安装环境中的 `cann_ops_transformer.ops` |
| CANN/驱动/HCCL | 设备任务执行、卡间通信、底层资源和错误上报 | 不在本仓完成 |

本仓无法单独保证外部包的 ABI、支持形状、kernel 名称和工作区字节数；最终要与部署环境版本配套。

## 3. 启动配置与静态校验

本章说明如何设置融合开关 `enable_fused_mc2`：从用户配置出发，经内部值转换、模型配置校验和功能互斥检查，得到最终开关值（0：关闭融合；1：允许融合）。第 4 章再决定本轮执行选择哪种通信方式。

```text
用户配置 enable_fused_mc2
    │
    ├─ 0：关闭融合
    ├─ 1：允许旧融合，MegaMoE 标志=False
    └─ 2：内部转换为1，根据算子包是否可发现设置 MegaMoE 标志
              ↓
derive_and_validate()
    当内部开关为1，且 MegaMoE 标志=True 时：
        检查 H、I、EP规模、专家数、Top-K、量化配置等
        （具体检查项随硬件不同）
              │
              ├─ 通过：保留 enable_fused_mc2=1
              └─ 不通过：设为0，并打印 warning
```

此外，`derive_and_validate()` 还处理融合与其他功能的互斥关系，例如关闭共享专家多流重叠、拒绝与分层通信组合。

### 3.1 用户配置与内部值

| 用户配置 `enable_fused_mc2` | 内部值 | `_MEGA_MOE_SUPPORTED` |
| --- | --- | --- |
| 0：关闭融合 | 0 | False |
| 1：允许旧融合 | 1 | False |
| 2：尝试 MegaMoE | 1 | Python 能找到 `cann_ops_transformer` 算子包时为 True，否则为 False |

`is_mega_moe_supported()` 读取上述标志，仅表示包可发现且配置允许尝试 MegaMoE，不代表算子已验证可运行。

代码：[vllm_ascend/ascend_config.py](../../vllm_ascend/ascend_config.py)，`_validate_user_input_ranges()`。

### 3.2 模型配置校验与融合禁用

`derive_and_validate()` 在内部开关为 1 且 MegaMoE 标志为 True 时，调用 `_is_megamoe_supported_by_config()` 检查下表条件：**通过则保留 1；不通过则改为 0 并 warning，MegaMoE 标志不变**。H 或 I 无法获取也视为不通过。

| 检查项 | A5 的通过条件 | 非 A5 路径（如 A3）的通过条件 |
| --- | --- | --- |
| 专家输入维度 H | 在 {1024,2048,3072,4096,5120,6144,7168,8192} 中；Kimi K3 额外允许 3584 | 1024～8192，且是 512 的整数倍 |
| 专家中间维度 I | 检查 **2I** 是否在 {1024,2048,3072,4096,7168} 中；MiniMax M3、Kimi K3 额外允许 6144 | 检查 **I 本身**是否在 1024～3072，且是 512 的整数倍 |
| 推导的 EP 规模 | 至少为 2，且专家总数能被它整除；计算参数见下文 | 此函数不检查 |
| Top-K | 1～32 | 此函数不检查 |
| 配置中的量化类型 | 此函数没有完整校验实际量化类型 | 读取到的量化名称须为 w8a8、w4a8、w8a8_dynamic、w4a8_dynamic、quanttype.w8a8 或 quanttype.w4a8；未读取到则放行 |

模型参数与部署参数的对应关系：

| 检查项 | 参数来源 | 含义 |
| --- | --- | --- |
| H | `hidden_size` | 每个 token 输入专家时的向量长度 |
| I | `moe_intermediate_size` | 单个路由专家的 FFN 中间维度 |
| 2I | `2 * moe_intermediate_size`，通常无独立配置字段 | gate、up 两个投影合并后的输出维度 |
| Top-K | 模型配置 `num_experts_per_tok`；缺失时取 `top_k_experts`，默认 1；Kimi K3 优先取 `num_experts_per_token` | 每个 token 选择的路由专家数；A5 此处限制为 1～32 |
| EP 规模 | 部署参数 TP、PCP、DP 计算得到，见下方公式 | 分布专家的 rank 数；A5 此处要求多 rank 且专家均匀分配 |
| 专家总数 E | `model_config.get_num_experts()` 从模型配置解析 | 路由专家总数；需满足 `E % EP == 0` |

EP 规模不是独立的模型字段，`world_size_across_dp` 也是计算属性。常规启动路径的计算为：

```text
TP  = tensor_parallel_size           ← --tensor-parallel-size
DP  = data_parallel_size             ← --data-parallel-size
PP  = pipeline_parallel_size         ← --pipeline-parallel-size
PCP = prefill_context_parallel_size  ← --prefill-context-parallel-size

world_size           = TP × PCP × PP
world_size_across_dp = world_size × DP
此处推导的 EP 规模    = world_size_across_dp ÷ PP
                     = TP × PCP × DP
```

PCP=1 时，简化为 `EP = TP × DP`。`--enable-expert-parallel` 是启用开关，不直接指定规模。上述公式对应常规启动路径；`external_launcher` 对 `world_size` 另有处理，应以最终配置和实际通信组为准。

以门控 FFN 专家为例：

```text
一个 token 的输入：[H]
           │
     ┌─────┴─────┐
     ▼           ▼
 gate 投影     up 投影
   H → I        H → I
     │           │
   激活 ──相乘───┘
           │
          [I]
           │
       down 投影
         I → H
           │
          [H]
```

gate、up 可合并为一次 `H → 2I` 投影，再拆成两路进行激活和相乘。

字段特例：Kimi K3 的 H 优先取 `routed_expert_hidden_size`；I 缺失时，非 A5 路径尝试 `intermediate_size`，A5 仅对 MiniMax M3 做此回退。

融合关闭后，后续选择 MC2、ALLGATHER 或 ALLTOALL，不回退到旧融合算子。

**检查范围**：仅校验静态配置，不保证权重布局和外部算子接口兼容，也不检查当前 batch 的 token 容量（见第 9.3 节）。

代码：[vllm_ascend/ascend_config.py](../../vllm_ascend/ascend_config.py)，`derive_and_validate()` 876 行、`_is_megamoe_supported_by_config()` 1080 行、`_is_a5_megamoe_supported_by_config()` 1118 行附近。

### 3.3 互斥与旁路影响

| 条件 | 代码行为 |
| --- | --- |
| 融合路径与 `multistream_overlap_shared_expert` 同时开启 | 配置阶段关闭共享专家多流重叠，并 warning |
| `mc2_comm_alg == hierarchy` 且开启融合 | 配置报错，不允许组合 |
| LoRA + 多 rank EP | 通用选择器提前返回 ALLTOALL，避开融合路径 |
| `mix_placement` 与共享专家 DP 或多流重叠组合 | 配置检查报错 |
| DeepSeek V4.1 在 A5 packed-cache 硬件路径 | 平台校验要求 Model Runner V2 |

配置互斥见 [vllm_ascend/ascend_config.py](../../vllm_ascend/ascend_config.py) 860、1004、1074 行附近；Runner 校验见 [vllm_ascend/platform.py](../../vllm_ascend/platform.py) 的 `_validate_model_runner_config()`，1047 行。

## 4. 运行时选择：何时使用 MegaMoE

第 3 章确定融合开关是否保留；本章说明运行时如何选择通信方式，以及怎样确认会调用 MegaMoE。

每次模型前向执行（一个 batch 的一次 forward）前统一选择通信类型，各 MoE 层沿用该类型并分别执行本层通信，后续轮次重新判断。

### 4.1 第一层：通用选择优先级

入口：[vllm_ascend/ascend_forward_context.py](../../vllm_ascend/ascend_forward_context.py)，`select_moe_comm_method()`，373 行。

| 顺序 | 条件 | 结果 |
| --- | --- | --- |
| 1 | 非 MoE | None |
| 2 | 未开 EP 或 EP 组大小为 1 | ALLGATHER |
| 3 | 配置 LoRA 且开启 EP | ALLTOALL |
| 4 | 硬件固定 ALLGATHER 策略 | ALLGATHER |
| 5 | 其余 | 调用硬件策略函数 |

返回的是枚举，尚未构造本次算子参数，也没有执行通信。

#### 4.1.1 A3(FUSED_OR_CAPACITY)：优先融合

`_select_fused_or_capacity_moe_comm_method()`，318 行附近：

```text
use_cann_megamoe() 为真 → FUSED_MC2
否则，内部开关为 1 且 EP <= 32 → FUSED_MC2（旧融合路径）
否则，T 为 None 或 T <= C → MC2
否则 → ALLTOALL
```

前两个融合分支没有比较当前 token 数与 C。超过输入容量会在之后的调用入口失败，而不是在这里自动选 ALLTOALL。

#### 4.1.2 A5(CAPACITY_AND_WORLD_SIZE)：选中 FUSED_MC2 的条件

同时满足以下三个条件就返回 `FUSED_MC2`：

| 条件 | 含义 | 如何得到 |
| --- | --- | --- |
| `enable_fused_mc2 == 1` | 融合开关仍然开启 | 用户配置 `enable_fused_mc2=2`，且通过模型配置校验 |
| `is_mega_moe_supported()` 为真 | 允许尝试使用 MegaMoE | 用户配置 `enable_fused_mc2=2`，且 Python 能找到 `cann_ops_transformer` 算子包 |
| `not is_draft_model`，或草稿量化类型受支持 | 主模型直接通过此项；草稿模型须检查量化类型 | 当前执行主模型，或草稿 MoE 使用 `W4A4MXFP`、`W4A8MXFP`、`W8A8MXFP` 中的一种 |

上述三种量化分别表示权重/激活采用 MXFP4/4、MXFP4/8、MXFP8/8。草稿模型未量化或类型不受支持时，跳过融合分支。能找到算子包不代表算子已验证可用；主模型通过第三项也不代表任意量化格式都受支持。

未选中融合时：`T` 为 None 或 `T <= C`，且 `world_size_across_dp > 1`，选择 MC2；否则，`world_size_across_dp <= Top-K` 时选择 ALLGATHER，其余选择 ALLTOALL。

**边界**：当前原始分支的上述融合条件不检查 batch 是否超容量；选中 `FUSED_MC2` 也不等于最终调用 MegaMoE，还需第 4.2 节的 `use_mega_moe` 为真。草稿保护依赖调用方正确传参，V2 路径差异见第 6.3 节。

代码：[vllm_ascend/ascend_forward_context.py](../../vllm_ascend/ascend_forward_context.py)，`_select_capacity_and_world_size_moe_comm_method()`、`set_ascend_forward_context()`；量化集合见 [vllm_ascend/quantization/quant_type.py](../../vllm_ascend/quantization/quant_type.py)。

#### 4.1.3 A2(CAPACITY_AND_EXPERT_DENSITY)：不选融合

`_select_capacity_and_expert_density_moe_comm_method()`：当每卡专家数<=24、推导 EP>=16、T 未超 C（或 None）同时满足时选 MC2，否则 ALLGATHER。

### 4.2 MegaMoE 使用标志

`use_cann_megamoe()` 在以下条件全部满足时返回 True：

| 条件 | 与第 3 章的关系 |
| --- | --- |
| `enable_fused_mc2 == 1` | 读取第 3 章处理后的最终结果 |
| `is_mega_moe_supported()` | 再次读取 MegaMoE 标志 |
| 硬件支持 `CANN_MEGAMOE` | 检查硬件能力（根据设备类型取得 hardware profile，判断能力集合中是否包含 `CANN_MEGAMOE`。当前 A3、A5 包含，A2、310P 不包含） |
| 模型是 MoE、开启 EP、无 LoRA | 执行路径的适用条件 |
| `1 < 实际 EP group.world_size <= 64` | 检查实际通信组规模；第 3 章 A5 检查的是配置推导规模，且未检查这个上限 |

最终调用 MegaMoE，需要同时选中 `FUSED_MC2` 且 `use_mega_moe=True`（根据 `FUSED_MC2` 选择融合执行类 `FusedMC2CommImpl`，该类再根据 `use_mega_moe` 选择算子）。具体代码见 [vllm_ascend/ops/fused_moe/moe_comm_method.py](../../vllm_ascend/ops/fused_moe/moe_comm_method.py) 的 `setup_moe_comm_method()`（73 行）和 `FusedMC2CommImpl.fused_experts()`（580 行）。

代码：[vllm_ascend/ascend_forward_context.py](../../vllm_ascend/ascend_forward_context.py) 的 `use_cann_megamoe()`；硬件能力声明见 [vllm_ascend/device/hardware_profile.py](../../vllm_ascend/device/hardware_profile.py)。

## 5. 原始分支与三个 PR 适配分支的对比

对比原始分支 `rfc/vllm_cann`（`7e37e29a7`）与适配分支 `integrate/dsnext-megamoe-vllm-cann`（`87b09d837`，三个 PR＋最终适配提交），当前缺少的主要是共享专家融合、A5 容量提前判断，以及空闲 DP rank 的协调优化。

### 5.1 适配分支补充的功能

| 功能 | 三个 PR 适配后的分支 | 当前原始分支 |
| --- | --- | --- |
| 共享专家融合进 MegaMoE | 将共享专家权重传入 MegaMoE，一次返回路由专家与共享专家的合并结果 | 只向 MegaMoE 传入路由专家权重，共享专家独立计算 |
| 共享专家融合的量化权重适配 | 路由专家支持 W4A8MXFP，共享专家可为 W4A8MXFP 或 W8A8MXFP；分别传递权重类型和 scale | 没有共享权重传入 MegaMoE 的链路 |
| 选择融合前检查输入容量 | A5 当前 token 数不超过容量才选 FUSED_MC2，否则继续选择普通路径 | 融合分支不判断 token 容量；选中后在执行入口检查，超限抛异常 |
| 按 PD 角色统一 A5 输入容量 | 根据预填充、解码、混合部署角色计算容量，选择、buffer 分配和执行检查使用同一规则；无 KV 配置按混合部署处理 | 已有 P/D 容量处理，但没有适配分支这套统一规则 |
| 空闲 DP rank 不干扰真实 batch | 区分真实任务与 dummy rank；以真实任务决定 token 数和图模式，空闲 rank 跟随 | V1 同步时未区分真实任务与 dummy，空闲 rank 可能放大 padding 或影响图模式 |

输入容量检查对应两处代码：

- **选择阶段未检查容量**：[vllm_ascend/ascend_forward_context.py](../../vllm_ascend/ascend_forward_context.py) 的 `_select_capacity_and_world_size_moe_comm_method()`（336～352 行），融合条件通过后直接返回 `FUSED_MC2`，没有比较 `num_tokens` 与容量。
- **执行阶段超限抛异常**：[vllm_ascend/ops/fused_moe/moe_comm_method.py](../../vllm_ascend/ops/fused_moe/moe_comm_method.py) 的 `_apply_cann_mega_moe()`（478～485 行），将 `hidden_states.shape[0]` 与 `token_dispatcher.max_num_tokens_per_rank` 比较，超限则抛出 `ValueError`，不自动切换后端。

对应 PR：

- [!18：A5 MegaMoE 接入](https://gitcode.com/xujiuxu9/vllm-ascend-dsnext/merge_requests/18)：引入容量控制和多 rank 协调相关修改。
- [!31：混合 PD 容量修复](https://gitcode.com/xujiuxu9/vllm-ascend-dsnext/merge_requests/31)：补充无 KV 配置时的容量处理。
- [!51：共享专家融合](https://gitcode.com/xujiuxu9/vllm-ascend-dsnext/merge_requests/51)：共享权重进入 MegaMoE；最终适配提交补齐接口校验、权重布局检查和安全回退。

共享专家融合的限制：外部算子必须提供共享权重接口；当前适配限定单个共享专家、兼容的激活和布局，TP>1 时共享权重需完整复制，并避开动态 EPLB、冗余专家和 `mix_placement` 等组合。不满足时仍独立计算共享专家。

基础 MegaMoE 接入、MXFP 路由权重处理、对称 buffer、通信组恢复等，当前原始分支已经具备。适配分支也没有解决 A5 `2I=4608` 的形状限制；新增路径此前仅完成主机回归测试，尚未通过 NPU 端到端验证。

### 5.2 已有能力的实现差异

能力有重叠，但三个原始 PR 与当前原始分支的实现不完全一致；最终适配时主要复用了当前分支的实现。

| 部分 | 三个 PR 的原始实现 | 当前原始分支实现 | 最终适配方式 |
| --- | --- | --- | --- |
| 基础 MegaMoE 接入 | 新增独立 MegaMoEBackend | 集成在 FusedMC2CommImpl 中 | 复用当前执行类，补充缺失功能 |
| MXFP 路由权重 | 主要针对 W4A8MXFP；执行时转置，使用堆叠权重并检查布局 | 加载后准备 MegaMoE 专用权重和 scale，通过量化方法提供；支持多种 MXFP 类型 | 保留当前权重处理，仅补充共享权重传递 |
| 对称 buffer | 缓存在专用通信组上，用包含形状、容量、量化信息的 key 校验复用 | 保存在融合执行对象中，懒分配；分别处理输入容量和 dispatch 后接收容量 | 保留当前 buffer 框架，补充 A5 容量计算 |
| 通信组 | 新建独立 mega_moe 组，提前初始化 HCCL | 使用已有 MC2 通信组 | 保留当前 MC2 通信组 |
| 通信组恢复 | 原始 PR 中未见当前这套 teardown/restore 重绑定机制 | 销毁前标记失效，恢复后通过 update_group() 重绑定 MegaMoE 上下文 | 完整保留当前恢复机制 |

适配思路：使用当前分支已有的 MegaMoE、权重和通信基础设施，再补入共享专家融合、容量提前判断等差异功能，避免维护两套后端。

历史适配说明见 [dsnext-vllm-cann-adaptation.md](dsnext-vllm-cann-adaptation.md)，其中旧基线和提交号不作为本文当前快照。

## 6. 初始化、加载、profiling 与正式执行顺序

### 6.1 初始化时序

```text
创建并校验 VllmConfig / AscendConfig
    ↓
建立 Worker，init_device，建立 TP/DP/EP/MC2 通信组
    ↓
创建 Model Runner V1 或 V2
    ↓
set_mc2_tokens_capacity()；set_mc2_mask()
    ↓
构造模型/MoE 工厂、Router、RoutedExperts、AscendMoERunner
    ↓
setup_moe_comm_method(moe_config)
    ├─ 创建 ALLGATHER / ALLTOALL / MC2 / FUSED_MC2 对象（EP>1）
    └─ FusedMC2CommImpl 初始化时加载 MegaMoE 包/扩展并绑定激活 ABI
    ↓
加载 checkpoint；各量化方法 process_weights_after_loading()
    ↓
profile_run / dummy run / 图预热或捕获
    └─ 第一次真正调用 MegaMoE 时懒分配对称 buffer
    ↓
正式请求：选择本步通信方式 → 准备输入 → 路由 → 专家 → finalize
```

Model Runner 与 MoE Runner 不同：前者组织整个模型；后者组织模型中的一个 MoE 层。

### 6.2 V2 profiling 的额外步骤

[vllm_ascend/worker/v2/model_runner.py](../../vllm_ascend/worker/v2/model_runner.py)：

- `__init__` 258 行附近初始化 MC2 容量和保留 mask。
- `profile_run()`，584 行：在 profile 最大 token 数大于 MC2 容量、容量对应策略为 MC2/FUSED_MC2 时，先临时禁用编译做一次容量大小的 `_dummy_run()`，预留通信资源。
- 标准 profiling 包在 `override_mrv2_in_profile_run(True)` 中，RoutedExperts 用这个标记做负载均衡的假路由。

这说明启动时已经可能调用 MegaMoE、分配 buffer。看到一次调用日志不能只归因于用户请求。

### 6.3 V1 与 V2 的 forward context 入口不同

| 路径 | 入口 | 行为 |
| --- | --- | --- |
| V1 | `set_ascend_forward_context()` | 从 DP token 信息选择策略，记录 draft 标记、mask、padding、use_mega_moe |
| V2 | `NPUPlatform.set_additional_forward_context()` | 上游构造 context 时补 Ascend 字段；使用 DP 最大 token 数保持策略一致 |

V2 入口在 [vllm_ascend/platform.py](../../vllm_ascend/platform.py) 643 行附近。当前该函数把 `is_draft_model=False` 写成临时默认，调用 selector 时也没传草稿量化参数。V1 的 context 管理器则显式处理这些参数。

所以“selector 有草稿量化回退”不等于“所有 V2/DSpark 调用都必然经过带正确 draft 标记的回退”。定位草稿行为需继续看实际 forward context 的建立路径。

### 6.4 完整路由专家调用链

```text
AscendMoERunner.forward()
  → self._forward_entry(...)                   [上游注册的 MoE 自定义入口]
  → AscendMoERunner._forward_impl()
  → AscendRoutedExperts.forward_impl()
      → get_moe_comm_method(type, moe_config)
      → comm.prepare(...)
      → self._select_experts(...)
          → self.router._select_experts(...)
      → self.quant_method.apply(...)           [量化适配器]
          → scheme.apply(...)                 [具体 W4A8MXFP 等方法]
              → build_fused_experts_input(...)
              → comm.fused_experts(..., quant_method=scheme)
                  → scheme.get_fused_mc2_weights(layer)
                  → comm._apply_cann_mega_moe(...)
                      → self.mega_moe(...)
      → 统计 EPLB expert_tokens（按需）
      → comm.finalize(...)
```

非量化方法没有同样的量化 wrapper，但仍使用统一输入和通信对象。

## 7. MoE 输入和路由参数怎样算出来

定义：T 是进入本地专家调用的 token 行数，H 是路由专家输入维度，I 是每个专家的单分支中间维度，E 是全局专家数，L 是本 rank 专家数，K 是每个 token 选中的专家数。

从模型配置到 MoE 层的主要标量传递如下：

| MoE 标量 | DeepSeek V4.1 模型来源 | 后续使用 |
| --- | --- | --- |
| H | config.hidden_size；latent 模型可先投影为 routed hidden | Gate 输入、专家权重输入、symm buffer hidden |
| I | config.moe_intermediate_size | 工厂配置 intermediate_size，再推导 intermediate_size_per_partition |
| E | config.n_routed_experts，物理部署还要计入冗余/混合专家 | Router 全局专家范围、本地专家分配、buffer num_experts |
| K | config.num_experts_per_tok | Router Top-K、buffer num_topk、ID/prob 的列数 |
| 共享 MLP 中间维度 | moe_intermediate_size * n_shared_experts | 独立共享专家 MLP，不直接作为路由专家的 I |
| 激活/缩放 | hidden_act、swiglu_limit、routed_scaling_factor 等 | 激活 ABI、路由或输出缩放 |

本仓工厂适配见 [vllm_ascend/patch/platform/patch_fused_moe.py](../../vllm_ascend/patch/platform/patch_fused_moe.py) 的 `_ascend_FusedMoE()`：构造 Ascend Router，协调 EPLB/冗余专家配置，再把参数传给上游工厂，并以注册机制得到 Ascend Runner/专家对象。

EP 下上游 MoE 并行配置通常把“单专家矩阵的 TP 大小”改为 1，因为专家以完整矩阵部署到不同 EP rank；这不等于物理 TP 组不存在。共享专家线性层、token 切片和 finalize 仍可能使用物理 TP 组。因此不能总用 `I/启动参数TP` 推测每张卡的专家矩阵尺寸，应读取实际 moe_config.intermediate_size_per_partition。

### 7.1 数学含义

对第 t 个 token，普通 SwiGLU 路由专家可以写成：

```text
scores[t] = Gate(x[t])                            [E]
ids[t], probs[t] = Router(scores[t], ...)          [K], [K]
z_e = SiLU(GateProj_e(x[t])) * UpProj_e(x[t])      [I]
y[t] = Σ_e probs[t,e] * DownProj_e(z_e)            [H]
```

真实模型可能使用 sqrtsoftplus、hash/视觉路由、SiTU、输出缩放、latent 投影和不同的 Top-K 归一化；公式用于解释尺寸，不替代具体模型语义。

### 7.2 Gate 和 Top-K 不在 MegaMoE API 里计算

`AscendMoERunner._compute_router_logits()` 经 `DeviceOperator.compute_gate_logits()` 计算内部 gate。A5 适配器可以融合输入 bf16→fp32 cast 与 gate matmul；其他设备可复用预转换的 FP32 输入和权重。

Router 产生 `[T,K]` 的 IDs 和概率，之后才进入 MegaMoE。MegaMoE 收到的是选择结果，不是 `[T,E]` 全量 logits。

路由入口：

- [vllm_ascend/ops/fused_moe/fused_moe.py](../../vllm_ascend/ops/fused_moe/fused_moe.py)：`_prepare_router_input`、`_apply_router_linear`、`_compute_router_logits`。
- [vllm_ascend/ops/fused_moe/router/fused_topk_router.py](../../vllm_ascend/ops/fused_moe/router/fused_topk_router.py)：`AscendFusedTopKRouter._compute_routing()`。
- [vllm_ascend/ops/fused_moe/router/grouped_topk_router.py](../../vllm_ascend/ops/fused_moe/router/grouped_topk_router.py)：普通 softmax/sigmoid/sqrtsoftplus、grouped Top-K 的回退实现。

### 7.3 DeepSeek V4.1 的路由特例

[vllm_ascend/models/deepseek_v41/model.py](../../vllm_ascend/models/deepseek_v41/model.py) 的 `DeepseekV41MoE`：

- Gate 是未量化 ReplicatedLinear，维度 H→E，并要求预转换 FP32 权重。
- hash 层构造 tid2eid；非 hash 层使用 e_score_correction_bias。
- 视觉场景还可能使用 bias_vl 和 image sentinel。
- Router 内部把 input_ids 按通信方法 pad/split 或 all-gather，保持与 logits 行数一致；padding ID -1 在相应路径被清理为 0。
- sqrtsoftplus 路径可以进入 `moe_gating_top_k_hash`；普通打分则经 DeviceOperator.moe_gating_top_k 或 native fallback。

Top-K 内部也有融合算子，但“Top-K fused”和“MegaMoE dispatch/MLP/combine fused”是不同层次，profile 中不能混淆。

### 7.4 缩放与负载均衡

`topk_weights` 是路由系数，不是模型训练权重。renormalize、routed_scaling_factor 可能在 router 中应用，也可能由模型/工厂要求在输出阶段应用。分析精度时要追踪实际工厂传参，不能额外乘一次默认缩放。

`AscendRoutedExperts._select_experts()` 在路由后还可能：

1. 用 log2phy 将逻辑专家 ID 映射为物理专家 ID。
2. mix_placement 时追加共享专家。
3. force EPLB 时修改专家选择。
4. profiling 时用随机均衡 ID 替代实际路由，避免假输入造成极端负载。

因此 profiling 的专家负载分布不能直接代表正式业务。

## 8. token 对齐：三种“大小”要分开

### 8.1 当前 token 数、TP 对齐数与本 rank 行数

设各 DP rank 的 token 数为 T_d，原始 TP 大小为 P：

```text
Tmax = max_d(T_d)
Tpad = ceil(Tmax / P) * P
本 TP rank 专家输入行数通常为 Tpad / P
```

例如 DP 两路分别 5、9 个 token，TP=2：Tmax=9，Tpad=10，每个 TP rank 输入 5 行。有效位仍需区分真实 token 与补齐 token。

这只是 padding 算术示例，不代表前述模型支持矩阵允许任意 EP/模型组合。

### 8.2 MC2/FUSED_MC2 共享的 prepare

`FusedMC2CommImpl._get_prepare_finalize()` 返回 `PrepareAndFinalizeWithMC2`。

入口：[vllm_ascend/ops/fused_moe/prepare_finalize.py](../../vllm_ascend/ops/fused_moe/prepare_finalize.py)，252 行。

非 SP 输入：

1. 从 context 取 mc2_mask、padded_num_tokens。
2. 将 hidden_states 和 router_logits 用零行补到目标长度。
3. TP>1 时按 token 维切片，取当前 TP rank。
4. 对 mask 做对应切片。
5. 保存原 token 数和 padded shape 供 finalize 使用。

SP 已分片输入（replace_allreduce=True）：

1. 不再重复完整输入的 TP 切片。
2. 按原本地行数切出有效 mask。
3. 补到 `padded_num_tokens // TP`，新增 mask 位为 False。
4. finalize 裁剪回原本地行数。

`_pad_tokens_with_cat()` 在小 padding 时使用缓存零块加 cat，大 padding 时使用 F.pad；它是 token 行维补齐，不是专家特征维补齐。

### 8.3 不要把 token 对齐理解成 512 对齐

| 对齐/约束 | 含义 |
| --- | --- |
| ceil(T/TP)*TP | token 行数能按 TP 切分 |
| 图捕获 batch bucket | Runner 选定的图执行形状，可能进一步增加 token 行 |
| H/I 的 shape 白名单 | 矩阵特征维度约束 |
| INT4/FP4 打包 | 两个四位值使用一个 byte 等存储约束 |
| MX scale 配对 | group scale 被 reshape 为末尾 2 的布局 |

代码没有一个“所有 token 必须是 512 倍数”的统一规则。

### 8.4 finalize 做什么

非 SP 且 TP>1：创建新的输出 tensor，all-gather 本地 token 切片，再裁剪回原 token 数。不能复用可能与共享专家输入别名的旧 tensor。

SP：只裁剪回原本地 token 数。Runner 根据通信类型、SP 和输出变换决定后续归约，避免重复归约。代码注释提到的“输出已归约”要结合实际 gather/分片操作理解，不能把每一层一概叫作 all-reduce。

所以即使路由专家内部用了 MegaMoE，外层仍可能出现 AllGather；看到 AllGather 或 dispatch 字样不能直接判断融合未生效。

## 9. 容量：MC2 输入容量与 MegaMoE 接收容量

### 9.1 初始化一次的 MC2 容量 C

`set_mc2_tokens_capacity()` 在 [vllm_ascend/ascend_forward_context.py](../../vllm_ascend/ascend_forward_context.py) 243 行附近，进程级全局只初始化一次。

先选基础 token 上限 B：

| 条件，按顺序 | B |
| --- | --- |
| enable_prefill_mc2，或 MegaMoE 且非严格 decode-only 节点 | max_num_batched_tokens |
| 否则配置了图捕获 sizes | max_cudagraph_capture_size |
| 否则 | max_num_reqs * uniform_decode_query_len |

然后：

```text
C_rank = min(ceil(B / TP), 单 rank 路径上限)
C = C_rank * TP
```

| 路径 | 单 rank 上限 |
| --- | --- |
| use_cann_megamoe=True | 16384 |
| 开融合但不是 MegaMoE | 512 |
| 普通 MC2 | 512 |

`TokenDispatcherWithMC2.max_num_tokens_per_rank = C // TP`。这就是 `_apply_cann_mega_moe()` 在真正调用前使用的每 rank 输入检查值。

### 9.2 decode-only 的判断比 kv_consumer 更严格

`_is_decode_only_node()` 要求：KV 配置表明它是纯 consumer 或 DecodeBenchConnector，同时启用 recompute_scheduler_enable。

无 KV 配置、kv_both、consumer 但未开该 scheduler，均不能直接按严格 decode-only 对待。

### 9.3 选中融合后仍可能输入超容量

`_apply_cann_mega_moe()` 第一段检查：

```python
num_tokens = fused_experts_input.hidden_states.shape[0]
num_max_tokens = self.token_dispatcher.max_num_tokens_per_rank
if num_tokens > num_max_tokens:
    raise ValueError(...)
```

这里的 num_tokens 已是 prepare 后本 rank 行数；selector 中的 T 则通常是 DP-wide 最大 token 数，二者不能直接混用。

原始 A3/A5 融合 selector 没有以这项比较作自动回退；当前请求过大时是 fail-fast。不能描述为“所有超限场景自动走 ALLTOALL”。

### 9.4 接收容量 R：dispatch 后会有 token-expert 复制

定义 W 为 MC2 组大小，L 为每 rank 专家数，K 为 Top-K，C_rank 为每来源 rank 输入上限：

```text
R_safe = max(1, C_rank * W * min(K, L))
```

原因：最多 W 个来源，每个来源最多 C_rank 个 token，一个 token 最多在本 rank 的 min(K,L) 个专家上计算。

当前分配方式：

| 节点角色 | max_recv_token_num |
| --- | --- |
| 严格 decode-only | R_safe |
| P 或 PD 合设等 | min(mega_moe_max_tokens, R_safe) |

P/PD 下配置值是显存与负载容量的折中。代码明确 warning：实际分发后接收数超过容量可能发生精度退化。本仓没有在这里用真实负载自动扩容或安全回退。

### 9.5 容量数字示例

假设 W=8、K=6、E=384，因此 L=48，C_rank=1024：

```text
R_safe = 1024 * 8 * 6 = 49152
```

mega_moe_max_tokens=65536 时 P/PD 实际 R 为 49152；配置为 16384 时 R 为 16384，并不保证可以容纳所有最坏路由分布。

若 C_rank=16384，其他条件相同，则 R_safe=786432。decode-only 选择严格上界时可能带来很大内存压力；实际字节数还取决于 H/I、量化模式、临时资源及外部实现，不能用 R 直接当字节数。

## 10. 多 rank 协调、通信组与 buffer 生命周期

### 10.1 EP 与 MC2 组

[vllm_ascend/distributed/parallel_state.py](../../vllm_ascend/distributed/parallel_state.py) 153 行附近建立一个 EP-like 的独立 MC2 组。组成员由 DP、PCP、TP 网格组合，代码构造规模为 `DP*PCP*TP`，按 pipeline stage 等外层维度分组。

常见无 PCP 的单 stage 配置可以理解为 EP 规模=DP*TP。PP 不应额外乘入同一个 EP 组。实际执行的 rank_id、world_size 和 group_name 要以 get_mc2_group() 为准。

MC2 communicator 在 HCCL registry 中有独立 reuse domain，不能把它当成任意可替代的 TP communicator。

### 10.2 DP 为什么要统一策略和 token 形状

V2 的平台 hook 使用 `dp_metadata.num_tokens_across_dp_cpu.max()` 选通信方法，确保不同 DP rank 不会因本地 token 数不同选择不同后端。

V1 `_sync_metadata_across_dp()` 用 CPU group 的 all-reduce 汇总 token 数和图模式。MegaMoE 即使 eager 也要求统一输入时，会给各 rank 选择同一个最大 token 数。

[vllm_ascend/utils.py](../../vllm_ascend/utils.py) 的 `should_skip_allreduce_across_dp_group()` 在 hierarchy 或 MegaMoE 支持标志为真时直接返回 False。当前代码不允许依赖动态 global_bs 来跳过 MegaMoE 的 DP 协调。

注意原始分支 V1 只同步两项元数据，没有适配分支新增的 active/dummy rank 标记。不应声称它已经包含那项 idle-DP 优化。

### 10.3 TokenDispatcher.global_bs 的意义

`TokenDispatcherWithMC2.__init__()`：

```text
max_num_tokens_per_rank = C // TP
max_global_bs = max_num_tokens_per_rank * MC2_world_size
global_bs = max_global_bs（允许跳过 DP 同步时），否则 0
```

global_bs=0 表示当前统一模式；非零用于普通 MC2 的动态 rank token 场景。不要把 global_bs=0 解读为当前 batch 没有 token。

### 10.4 对称 buffer 必须使用 rank 不变参数

`_init_mega_moe_symm_buffer()` 的 num_max_tokens_per_rank 从以下静态量取：

- global_bs>0：global_bs//world_size。
- 否则：dispatcher.max_num_tokens_per_rank。

不能拿本次 hidden_states.shape[0] 来决定分配尺寸，否则各 rank 可能不同步分配、握手参数不一致，导致 HCCL 挂起或失败。

参数还包括 E、K、H、2I、接收容量和量化模式。所有 rank 不只需要输入容量相同，还需要这些模型/通信设置兼容，并按相同 collective 顺序进入首次分配。

实际分配接口是：

```python
self.get_symm_buffer_for_mega_moe(
    get_mc2_group().device_group,
    self.moe_config.num_experts,
    num_max_tokens_per_rank,
    self.moe_config.experts_per_token,
    hidden=self.moe_config.hidden_dim,
    intermediate_hidden=2 * self.moe_config.intermediate_size_per_partition,
    max_recv_token_num=max_recv_token_num,
    dispatch_quant_mode=dispatch_quant_mode,
    dispatch_quant_out_dtype=dispatch_quant_out_dtype,
)
```

前四个位置参数依次是通信组、全局专家数、每来源 rank 最大输入 token 数、每 token 专家数。这里的 intermediate_hidden 来自实际 MoE 层的 partition 维度；它不依赖这一批 token 的实际路由结果。

### 10.5 懒分配与复用

构造 FusedMC2CommImpl 时 `mega_moe_symm_buffer=None`；第一次 `_apply_cann_mega_moe()` 调 allocate，之后复用，并更新对象上的 dispatch quant mode/dtype。

这意味着首次调用比稳态调用多出集体资源握手与分配工作；直接以首次调用耗时代表稳态性能不合适。

### 10.6 sleep/wakeup 后重建 HCCL

[vllm_ascend/device_allocator/sleep_mem_optimized.py](../../vllm_ascend/device_allocator/sleep_mem_optimized.py) 的生命周期管理会遍历通信对象：

1. teardown 前 `prepare_hccl_teardown()` 标记上下文失效，并检查外部 `_ctx_manager.update_group()` 接口存在。
2. 重新建组后 dispatcher 刷新 communicator 名称。
3. `refresh_hccl_runtime_state()` 用新组名在 inference_mode 下更新 context。
4. 更新 group、rank_id、world_size、ccl_buffer_size，并执行 barrier。

只改 group_name 字符串不足以恢复外部算子持有的旧通信上下文。

### 10.7 进程内的缓存不是“每层一个永远独立实例”

`_MoECommMethods` 用 `(MoECommType, (num_experts,num_local_experts))` 为 key。它不是用层名、H、I、激活或量化作为完整 key。相同 key 的后续 setup 会覆盖前面的对象。

Runner 因而把 activation kwargs 保存到 owning routed layer，避免拿错后建层的激活参数。不同 H/I、quant、动态专家重排与多模型并存仍要注意对象所有权和 buffer 复用约束。

DeepSeek V4.1 target/draft 模型有 moe_comm_methods 快照，但其构造使用 `get_moe_comm_method(kind)`（没有 moe_config）。该调用查 `(0,0)` 形状 key；RoutedExperts 在实际执行时又按本层 moe_config 重新查找和写入 context。不能仅看模型快照就假设所有 communicator 都已经隔离正确，需追实际消费位置。

## 11. 权重：逻辑形状、存储形状和加载后布局

### 11.1 逻辑矩阵

只看一个本地专家，采用 out/in 表示法：

```text
W13（Gate+Up）：[2I, H]
W2（Down）：   [H, I]
```

L 个本地专家堆叠时是 `[L,2I,H]` 和 `[L,H,I]`。上游普通 GMM 可能要求 `[L,H,2I]`、`[L,I,H]`；MegaMoE 不一定使用相同方向。

I 在实际 buffer 中用 `moe_config.intermediate_size_per_partition`；配置校验用 hf_text_config 的全局 moe_intermediate_size。存在 TP/latent/padding 特例时，要分别核对，不应无条件认为两者数值相同。

### 11.2 量化 wrapper 与权重生产职责

`AscendFusedMoEMethod` 位于 [vllm_ascend/quantization/method_adapters.py](../../vllm_ascend/quantization/method_adapters.py) 263 行。

它持有真实 scheme，负责将上游 create_weights、process_weights_after_loading、apply 接到 scheme。scheme.apply 构造 `MoEFusedExpertsInput` 并把自身作为 quant_method 传入 communicator。

`FusedMC2CommImpl.fused_experts()` 再调用 `scheme.get_fused_mc2_weights(layer)`。布局选择主要由量化方法负责，communicator 不会现场替你把任意 checkpoint 转为正确权重格式。

### 11.3 A3 W8A8 INT

[vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py](../../vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py)，290、362 行：

- 原权重 transpose(1,2).contiguous()，转换为 GMM 所需方向。
- W8A8 权重转换为 FRACTAL_NZ。
- scale flatten，并为普通计算保留 FP32 scale。
- 融合路径通过 scale_from_float_to_int64 将 FP32 bit pattern 解释为 int32，再扩成 int64 保存；这不是把浮点 scale 直接数值取整。
- MegaMoE 使用逐专家 weight list 和 fused scale list；根据 EPLB 路径可能使用独立 clone，也可能是 unbind 视图。
- W8A8 MegaMoE 的 scale_bias 为 None；旧 dispatch_ffn_combine 使用占位空 bias 列表。

### 11.4 A3 W4A8 INT

[vllm_ascend/quantization/methods/w4a8/w4a8.py](../../vllm_ascend/quantization/methods/w4a8/w4a8.py)，348、373、480、656 行：

- checkpoint 格式不同，compressed-tensors 与 ModelSlim 有不同预处理。
- INT4 值打包为 int8 容器，一个 byte 装两个四位值。
- MegaMoE 使用这个 int8-packed 权重，并显式传 ACL INT4 权重类型。
- 非 MegaMoE GMM 分支会把四个 byte 重新 view 为 int32；不能把这个 int32-packed 形式直接传给 MegaMoE。
- scale 转换为算子约定的整型容器，按专家展开。
- w13_scale_bias / w2_scale_bias 是量化精度补偿项。ModelSlim 路径对相应维度求和，MegaMoE 路径按专家 flatten 并转 FP32；它们不是任意神经网络 bias。

本仓注释指出：错误地把 int32-packed W4A8 当作 MegaMoE 的 int8-packed 权重，会让算子推导出的 N 错误，导致 weight2 维度校验失败。相同逻辑矩阵不代表同样的物理 shape/dtype。

### 11.5 A5 W4A8MXFP

[vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py](../../vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py)，194～373 行。

checkpoint 分配形状：

| 张量 | 物理 shape | 容器 dtype |
| --- | --- | --- |
| w13_weight | [L,2I,H/2] | uint8，装 FP4 |
| w2_weight | [L,H,I/2] | uint8，装 FP4 |
| w13_weight_scale | [L,2I,H/G] | uint8，装 E8M0 |
| w2_weight_scale | [L,H,I/G] | uint8，装 E8M0 |

G 是 group_size，默认配置通常为 32，实际取 quant_description。

加载后 MegaMoE 分支：

1. scale reshape 为 `[L,N,(K/G)/2,2]`。
2. 保持原 out/in 方向，按专家 clone。
3. 权重做 FRACTAL_NZ，显式传 FP4 输入语义及 format-cast 所需 metadata。
4. scale 用 `.view(torch.float8_e8m0fnu)` 改成语义 dtype。这里是同一组 bits 的重解释，不是重新学习或数量量化。
5. 生成 `cann_mega_moe_w13_weight_list`、`cann_mega_moe_w2_weight_list` 及两组 scale list。
6. 对原参数 dispose_tensor，解除原参数对旧 storage 的持有，减少重复保存；实际回收还取决于其他引用。reload 与 EPLB 不能只假设原参数仍含完整数据。

普通分支则使用另一路格式变换，并 transpose 成 GMM 所需方向。后端切换必须同时考虑是否还保留那一路权重。

举例，H=5120、I=2304、L=48 时，加载时的 FP4 容器 shape 为：

```text
w13_weight：[48, 4608, 2560]
w2_weight： [48, 5120, 1152]
```

这里 2560/1152 是 byte 容器维度，逻辑输入维度仍为 5120/2304。这个例子只解释 checkpoint 布局；当前 A5 白名单仍会拒绝 2I=4608，不能因为权重能表示就认为融合算子可以执行。

### 11.6 A5 W4A4MXFP 与 W8A8MXFP

| 方法 | 与 W4A8MXFP 的主要差异 | 入口 |
| --- | --- | --- |
| W4A4MXFP | 权重仍 FP4；通信激活输出也为 FP4；scale group 数为奇数时先补一个条目再按 2 配对；部分 compressed-tensors 权重名有 _packed | [vllm_ascend/quantization/methods/w4a4/w4a4_mxfp4.py](../../vllm_ascend/quantization/methods/w4a4/w4a4_mxfp4.py)，229、320 行 |
| W8A8MXFP | 权重是 FP8，不除以 2；MegaMoE 分支保留原 out/in 方向并逐专家 clone，scale 同样 reshape/view E8M0；这段分支没有与 FP4 完全相同的 NZ cast 代码 | [vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py](../../vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py)，357、389 行 |

不要用“融合权重一律 NZ、一律 int8、一律先转置”概括所有分支。应按具体量化方法逐项确认。

### 11.7 量化在哪里发生

权重主要从已量化 checkpoint 加载，再做布局/scale 变换；不是每个 token 重新量化权重。

输入 hidden_states 经 prepare 后仍可为 BF16 等类型；MegaMoE 的 dispatch_quant_mode 与输出 dtype 控制算子内部通信/计算使用的量化格式。普通 MC2/GMM 路径则有独立动态量化与激活量化 hook。

选中 MegaMoE 后不会先走普通 apply_moe_mlp 再走融合 MLP，后者已在融合算子内完成。

## 12. 统一输入、实际算子入参和返回值

### 12.1 MoEFusedExpertsInput 是软件载荷，不等于底层 ABI

入口：[vllm_ascend/ops/fused_moe/dataclass/fused_experts.py](../../vllm_ascend/ops/fused_moe/dataclass/fused_experts.py)。

| 字段 | 来源与用途 |
| --- | --- |
| hidden_states | prepare 后 token 特征 |
| topk_ids / topk_weights | Router 与物理专家映射后的结果 |
| layer | owning RoutedExperts 对象，用于读取正确权重及激活 |
| weights | 统一 MoEWeights 载荷；主要执行路径由 scheme hook 提供，部分为旧路径兼容 |
| routing | expert_map、冗余专家数、mask、输入路由加权标志、per-token scale |
| quant | QuantType、MXFP activation/weight/scale 类型等 |
| activation / dynamic_eplb / lora_context | 通用 MoE 编排属性 |

不应认为载荷里每个字段都已被 MegaMoE 分支消费。比如独立 expert_map、global_redundant_expert_num、apply_router_weight_on_input 没有作为同名参数传入本次 MegaMoE 调用；等价行为是否已由 ID 预处理或算子实现承担，需要单独核对。

### 12.2 实际 MegaMoE API

入口：[vllm_ascend/ops/fused_moe/moe_comm_method.py](../../vllm_ascend/ops/fused_moe/moe_comm_method.py)，540 行：

```python
out, expert_tokens = self.mega_moe(
    hidden_states,
    topk_ids.to(torch.int32),
    topk_weights.to(torch.float32),
    weight1,
    weight2,
    self.mega_moe_symm_buffer,
    l1_weights_sf=weight_scales1,
    l2_weights_sf=weight_scales2,
    l1_bias=l1_bias,
    l2_bias=l2_bias,
    x_active_mask=x_active_mask,
    weight1_type=weight_type,
    weight2_type=weight_type,
    **activation_kwargs,
)
```

| 参数 | 当前含义/要求 |
| --- | --- |
| hidden_states | [T,H]，T 是当前 rank prepare 后行数；具体可接受 dtype 由算子包决定 |
| topk_ids | [T,K]，调用前转 int32 |
| topk_weights | [T,K]，调用前转 FP32 |
| weight1 / weight2 | hook 产生的本地专家列表；外层只把非 list 包成单元素 list，不自动按 E 拆开 |
| symm_buffer | 外部分配函数返回的通信/计算资源对象 |
| l1_weights_sf / l2_weights_sf | 按对应 INT/MXFP 约定准备的 scale |
| l1_bias / l2_bias | INT4 等路径的补偿项或 None |
| x_active_mask | A5 强制 None；A3 uniform 模式可传 contiguous int8 mask |
| weight1_type / weight2_type | ACL 类型枚举，决定 packed FP4/INT4 等的正确解释 |
| activation kwargs | 与外部 wrapper ABI 匹配的激活与标量参数 |

返回 out 继续经过 finalize；expert_tokens 用于专家负载记录/EPLB。MegaMoE 返回的新统计 tensor 才是本次结果，不能假设 self.expert_token_nums 始终是其输出存储。

### 12.3 A3/A5 mask 差异

- A5 MXFP：本仓明确传 None，不把保留的 mc2_mask 交给 MegaMoE。
- A3：global_bs==0 且有 mc2_mask 时，将其转 int8 并 contiguous 后传入。
- 非零 global_bs 时该调用分支不提供 mask；代码依赖普通 MC2 的动态 batch 场景区分。

A5 的 padding 行仍可能参与内部路由计算，后续裁剪才去掉对应输出；因此 padding 会影响接收容量和负载，不能只按真实 token 数估算 buffer。

## 13. 激活 ABI、外部包加载和模型兼容

入口：[vllm_ascend/ops/fused_moe/moe_utils.py](../../vllm_ascend/ops/fused_moe/moe_utils.py)，144、167 行。

### 13.1 加载阶段

`load_cann_mega_moe_ops()`：

1. 导入 cann_ops_transformer.ops。
2. 获取 get_symm_buffer_for_mega_moe 和 mega_moe。
3. 优先加载新目录的 mc2.common.comm_context 扩展。
4. 只有预期的模块不存在时，尝试旧目录 ops.comm_context，并调用 _ensure_loaded()。

这样将可能的 JIT 通信扩展构建提前到初始化，减少在 TorchDynamo/ACL graph 捕获中首次编译扩展的风险。包导入过程中可能加载其他算子，单个其他算子的 load warning 不能直接当作 MegaMoE 失败证据。

### 13.2 激活适配

| 激活 | 当前适配 |
| --- | --- |
| 标准 SwiGLU | 传 activation_clamp |
| SwiGLU-OAI / uninterleave | 检测 activation+activation_params，或 glu_alpha+glu_bias，或 swiglu_alpha+swiglu_beta |
| SiTU | 必须暴露 activation+activation_params，传 activation=situglu，beta 与可选 linear_beta |

通过 inspect.signature 检查显式参数；某些通用 wrapper 只能从 doc/schema 补充识别。缺少所需 ABI 时直接报错，而不是静默忽略激活参数。

每层保存 mega_moe_activation_kwargs，调用时优先取 owning layer 的值，避免共享 communicator 的后建层配置污染前一层。

## 14. 内存与 buffer 的完整视角

| 内存类别 | 生命周期/所有者 | 容易混淆之处 |
| --- | --- | --- |
| 专家权重/scale | 模型加载、量化方法；部分为列表 clone | 双布局或临时 clone 可提高加载峰值；view 不等于独立存储 |
| KV Cache | 引擎/attention cache 规划 | 与 MegaMoE symm buffer 无关，二者竞争同一设备内存 |
| mc2_mask | 进程内预留，长度 max_num_batched_tokens | 布尔有效 token mask，不是 KV 页表 |
| 普通 HCCL/MC2 buffer | 通信组/运行时 | 增加 HCCL_BUFFSIZE 不等价于扩大 MegaMoE 接收容量 |
| MegaMoE symm buffer | 外部包分配，通信对象持有并复用 | 配置由输入上限、接收上限、H/2I/K/E/quant 决定 |
| 中间激活/算子临时工作区 | forward / 算子执行 | 可能在 profiling、捕获或首调用才分配 |
| 图资源 | 图捕获/图管理器 | 多捕获形状可能增加常驻与峰值内存 |

本仓代码不足以推导外部对称 buffer 的准确字节公式。应通过包实现、分配日志和设备内存实测确认，不能只计算 T*H*元素字节数。

此前服务把 gpu-memory-utilization 从 0.9 降到 0.85 后恢复，结合 OOM 说明增加运行期余量有效；它不是单独证明 MegaMoE 是那次 OOM 的唯一分配者。

## 15. 从架构审查角度需要关注的边界

以下是源码可见的验证缺口或组合风险，不是宣称所有部署都会触发的已复现 bug。

| 边界 | 证据与影响 | 建议验证 |
| --- | --- | --- |
| A5 4608 被拒绝 | 配置白名单无该维度，直接关闭融合 | 对照外部算子实际支持后再决定适配，不能只删校验 |
| selector 与输入容量检查分离 | A3/A5 原始融合选择无 batch 容量条件，调用前才抛错 | 测试容量边界、图 padding 后 T、eager 大 prefill |
| A5 非量化 warning 不等于完整回退 | 加载阶段保留普通权重并 warning，但 apply 仍用 context 选择的 communicator；没有在该 warning 处修改本层 comm type | 跟踪 actual QuantType、draft 标记及调用对象，验证是否真正避开融合 |
| V2 草稿标记 | 平台 hook 暂设 False，selector 的 draft 条件不一定生效 | target/draft 分别检查 context、量化与缓冲区 |
| 量化布局加载后固定 | MXFP MegaMoE 分支 dispose 原 storage，非融合分支可能需要另一布局 | 测试运行时切换到 ALLGATHER/MC2 时，权重生产/消费是否仍配套 |
| communicator key 不完整 | key 只有类型和专家数，未含 H/I/quant | 异构 MoE 层、主草稿不同形状、在线 reload 的资源所有权检查 |
| 专家负载接收上限 | P/PD 的配置 R 可小于 R_safe | skew routing、Top-K 大、padding 多、负载均衡切换下精度测试 |
| EP 规模条件分散 | use_cann_megamoe 限制<=64，A5 selector 未重复限制 | EP 边界及 fallback 是否有可用旧算子路径 |
| mix_placement | IDs/K/物理专家布局会变化，MegaMoE 无独立 shared ABI | 核对实际 K、专家总数、buffer topk、路由缩放和共享输出 |
| 参数存在不等于已消费 | 通用 routing 载荷比 MegaMoE ABI 更宽 | 输入加权、冗余专家、expert_map 等特殊模型语义对照 |
| 权重热更新 | 外部列表 clone、dispose、restore 路径需要保持一致 | 更新后比较每专家参数、scale、输出与参考路径 |

对任何 fallback 的审查都应同时回答：控制流换了什么后端、该后端需要什么权重布局、布局现在是否仍存在、输出归约与缩放是否仍一致。

## 16. 现有测试能证明什么

| 测试 | 覆盖方向 | 限制 |
| --- | --- | --- |
| [tests/ut/ops/test_moe_comm_method.py](../../tests/ut/ops/test_moe_comm_method.py) | P/D 接收容量、clamp、激活 kwargs、分配参数与调用载荷 | mock 为主，不执行完整 NPU 服务 |
| [tests/ut/test_ascend_forward_context.py](../../tests/ut/test_ascend_forward_context.py) | MC2 容量、decode-only、硬件策略分发 | 不能替代设备通信正确性 |
| [tests/ut/quantization/test_method_adapters.py](../../tests/ut/quantization/test_method_adapters.py) | wrapper 权重创建/加载/调用委托、EPLB 视图 | 不是所有量化形状的算子精度矩阵 |
| [tests/e2e/nightly/single_node/ops/test_mega_moe.py](../../tests/e2e/nightly/single_node/ops/test_mega_moe.py) | 八卡 A5、W4A8MXFP、SiTU/SILU、生产权重预处理、融合与参考路径和图执行对照 | 测试明确要求八张空闲 A5 和支持 SiTU 的包；不等于当前环境已执行 |

单算子 e2e 默认使用 H=3584、I=3072、E=224、K=16，模拟 Kimi K3 的本地专家规模；它不覆盖 DeepSeek V4.1 的 I=2304/2I=4608，也不覆盖完整模型全部模块。

本次文档工作不改生产代码，不新增模拟功能测试，也不宣称 NPU 测试已通过。

## 17. 如何在部署现场确认走的是哪条路径

### 17.1 先确认实际加载的代码与包

```bash
python -c "import vllm_ascend; print(vllm_ascend.__file__)"
python -c "import cann_ops_transformer.ops as ops; print(ops.__file__); print(ops.mega_moe)"
```

`__file__` 前后各两个下划线。能打印接口不代表执行成功，但可先排除路径与缺包问题。

### 17.2 看四个关键观察点

| 观察点 | 需要记录 |
| --- | --- |
| 配置归一化后 | enable_fused_mc2、is_mega_moe_supported、模型 shape 校验结果 |
| selector 返回 | policy、method、当前 T、C、is_draft_model、draft quant |
| layer forward | 实际 communicator 类、实际 scheme QuantType、本地 X/IDs/probs shape |
| _apply_cann_mega_moe | per-rank 输入上限、symm 分配参数、实际 kernel 调用 |

DEBUG 可用于一次性定位；正式吞吐/时延采集恢复 INFO。调用处的一次性 INFO 只能证明曾进入 Python 分支，图重放阶段不一定再打印。

### 17.3 Profile 判据

Profiler 要采正式请求，并查看完整计算时间线和设备 kernel，不只看通信表。Python mega_moe 名称不一定与设备 kernel 同名；融合内部可有多个任务，外层 finalize 也可有 AllGather。

若怀疑未生效，先看配置白名单 warning，再看 method/use_mega_moe 双标志，再核对实际调用。不能仅凭有 dispatch、没搜到 megamoe 字符串就下结论。

### 17.4 多 rank 卡住时的优先顺序

先查所有 rank 最早异常，再对比进度和 worker 栈；重点确认首次 symm 分配参数及 collective 顺序一致。OOM、错误布局或参数校验失败可能使其他 rank 等待并最终出现 HCCL timeout。

CPU 侧 shm_broadcast 等待通常只是 EngineCore/Worker 消息未就绪的表象；不能直接据此判断 /dev/shm 太小。device 5 报错也不等于 device 5 是根因。

## 18. 按问题定位代码的索引

| 要回答的问题 | 文件与函数 |
| --- | --- |
| 为什么设置 2，运行时却检查 1 | [vllm_ascend/ascend_config.py](../../vllm_ascend/ascend_config.py)：_validate_user_input_ranges |
| 为什么 4608 被拒绝 | 同文件：_is_a5_megamoe_supported_by_config |
| A3/A5 怎么选通信方式 | [vllm_ascend/ascend_forward_context.py](../../vllm_ascend/ascend_forward_context.py)：select_moe_comm_method、三个 _select 函数 |
| MegaMoE 使用标志从哪来 | 同文件：use_cann_megamoe |
| 容量从哪来 | 同文件：set_mc2_tokens_capacity；[vllm_ascend/ops/fused_moe/token_dispatcher.py](../../vllm_ascend/ops/fused_moe/token_dispatcher.py)：TokenDispatcherWithMC2.__init__ |
| V2 如何保持 DP 选择一致 | [vllm_ascend/platform.py](../../vllm_ascend/platform.py)：set_additional_forward_context |
| 为什么还有共享 MLP | [vllm_ascend/ops/fused_moe/fused_moe.py](../../vllm_ascend/ops/fused_moe/fused_moe.py)：_forward_impl；[vllm_ascend/ops/fused_moe/shared_experts.py](../../vllm_ascend/ops/fused_moe/shared_experts.py)：forward |
| Router 的 Top-K、hash、视觉选择 | [vllm_ascend/ops/fused_moe/router/fused_topk_router.py](../../vllm_ascend/ops/fused_moe/router/fused_topk_router.py)：_compute_routing |
| token pad/split 与输出 gather | [vllm_ascend/ops/fused_moe/prepare_finalize.py](../../vllm_ascend/ops/fused_moe/prepare_finalize.py)：PrepareAndFinalizeWithMC2.prepare/finalize |
| weight list、scale、bias 怎么产生 | [vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py](../../vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py) 等：process_weights_after_loading、get_fused_mc2_weights |
| 实际 MegaMoE 调用和输入检查 | [vllm_ascend/ops/fused_moe/moe_comm_method.py](../../vllm_ascend/ops/fused_moe/moe_comm_method.py)：_apply_cann_mega_moe，469 行 |
| symm buffer 参数和集体分配 | 同文件：_init_mega_moe_symm_buffer，386 行 |
| 旧融合算子与 MegaMoE 分叉 | 同文件：FusedMC2CommImpl.fused_experts，563 行 |
| INT/MXFP 类型和激活 ABI | [vllm_ascend/ops/fused_moe/moe_utils.py](../../vllm_ascend/ops/fused_moe/moe_utils.py)：_get_cann_mega_moe_quant_settings、select_mega_moe_activation_kwargs |
| sleep/wakeup 的 HCCL 恢复 | [vllm_ascend/device_allocator/sleep_mem_optimized.py](../../vllm_ascend/device_allocator/sleep_mem_optimized.py) 与 FusedMC2CommImpl.refresh_hccl_runtime_state |
