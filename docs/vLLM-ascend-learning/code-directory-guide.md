# 按代码目录结构阅读 vLLM Ascend

这份索引回答两个问题：**这个目录负责什么，这个文件用于什么。** 已对照 README 的 DeepWiki 补充关键文件间的调用关系。原理、数据流和模块关系请配合 [详细设计](detailed-design.md) 阅读。

## 1. 版本与覆盖范围

- 扫描日期：2026-10-15；分支：`integrate/dsnext-megamoe-vllm-cann`。
- 提交：`203fbd7d43268cbca4ecb6b2a8a8aa4f43e57839`。
- 受 Git 管理的条目：**4462**，其中包含 **1** 个第三方子模块引用；不展开子模块内部文件。
- 额外收录当前已存在的 **10** 份未跟踪学习文档、PR 记录或操作脚本，并单独标注。
- 本索引共覆盖 **4472** 个条目，拆分为 **121** 个分册。索引自身不递归收录。
- 排除 `.git/`、`.ruff_cache/`、`__pycache__/`、构建产物以及独立旧副本 `v30-workspace/`。
- Python 文件通过 AST 提取类、函数和注释；C/C++ 提取可识别的注册、类型与接口；配置/文档读取名称、标题或关键项。
- 右栏引用保留源码中的原始名称/注释；历史注释可能滞后，实际启用状态以注册和配置验证为准。
- 核心执行文件补充了结合调用关系的职责说明。其他条目按声明、注释和工程角色归纳；右栏会注明仅按角色归类的情况。**这不是对每个文件逐行验证后的正确性审计。**

## 2. 先认识目录层次

| 目录 | 用一句话理解 |
| --- | --- |
| [vllm_ascend/](../../vllm_ascend) | Ascend 插件 Python 主包：将上游 vLLM 的引擎和模型接口接到 NPU 实现。 |
| [csrc/](../../csrc) | 自定义设备算子、PyTorch/CANN 绑定及算子构建工具链。 |
| [tests/](../../tests) | 单元、系统、集成、性能和测试基础设施；不是服务运行入口。 |
| [docs/](..) | 项目文档、站点构建、主题覆盖与当前学习资料。 |
| [examples/](../../examples) | 离线推理、分布式部署、权重更新和量化等参考使用示例。 |
| [benchmarks/](../../benchmarks) | 性能测量脚本、输入配置和报告模板。 |
| [tools/](../../tools) | 开发检查、调试、请求发送、文档生成与性能工具。 |
| [cmake/](../../cmake) | 仓库顶层 CMake 构建辅助。 |
| [.github/](../../.github) | GitHub 协作与自动化：CI、工作流、Issue/PR 模板和版本参考。 |
| [.agents/](../../.agents) | 开发辅助技能与操作说明，不参与模型前向执行。 |
| [.claude/](../../.claude) | 开发助手相关配置或说明，不属于推理运行代码。 |
| [.gemini/](../../.gemini) | 开发助手相关配置或说明，不属于推理运行代码。 |

文件类型也提供线索：`__init__.py` 是包边界/导出入口；`patch_*.py` 修改上游行为；`test_*.py` 验证行为；`op_host` 做主机侧注册和分块；`op_kernel` 做设备计算；`CMakeLists.txt` 组织构建。具体文件可能同时承担多项职责，应结合分册右栏阅读。

### 2.1 Python 主包的目录怎样配合

先用下面这张职责地图理解目录，再去查单个文件。实际目录树与全部文件表在后续章节；下表的“先读”表示学习顺序，不表示所有模型都走同一个实现。

| 目录/文件组 | 放在这里的原因 | 与其他目录的关系 |
| --- | --- | --- |
| `platform.py`、`ascend_config.py`、`envs.py`、`device/` | 把设备、用户选项与上游平台接口连接起来 | 为 Worker、后端和量化选择提供经过验证的配置 |
| `worker/`、`core/`、`ascend_forward_context.py` | 组织请求执行、调度扩展和缓存协议 | 把调度结果变成模型输入与设备坐标 |
| `models/` | 组合模型层、投影、专有模块和权重名称 | 调用 Attention、MoE、量化层与缓存接口 |
| `attention/` | 封装注意力后端、元数据构建和缓存读写约定 | 接收 Runner 坐标，调用模型相关算子 |
| `ops/`、`quantization/` | 实现基础层、MoE、格式适配和设备调用 | 一端接模型层，一端接 torch_npu、Triton、CANN 或本仓绑定 |
| `distributed/`、`eplb/` | 管理通信组、KV/权重传输和专家负载 | 支撑 TP/EP/CP、分离部署及缓存迁移 |
| `compilation/`、`worker/v2/aclgraph_utils.py` | 管理编译优化、图捕获与重放更新 | 约束输入地址、shape、workspace 和跨流依赖 |
| `spec_decode/`、`worker/v2/spec_decode/` | 把草稿生成和目标验证接入执行框架 | 依赖模型草稿结构、上下文缓存和拒绝采样 |
| `model_loader/`、`model_executor/`、`device_allocator/` | 加载、预取和管理参数/设备内存 | 与 KV 状态管理分工不同，启动时共同参与预算 |
| `patch/`、`_310p/` | 分别处理上游接口差异和 310P 专用实现 | 注册和配置决定是否启用，不应直接替换其他设备路径 |
| `lora/`、`ec_manager/`、`sample/` | 分别管理适配参数、encoder 结果复用和采样 | 按模型及请求配置接入主流程 |

### 2.2 平台接入：从安装声明读到 Worker

| 阅读顺序 | 文件 | 功能与后续关系 |
| --- | --- | --- |
| 1 | [setup.py](../../setup.py) | 查 entry points：上游从哪里发现平台和 general-plugin hook |
| 2 | [__init__.py](../../vllm_ascend/__init__.py) | 查 `register`、`_ensure_global_patch` 和各注册函数，确认初始化发生在哪个进程 |
| 3 | [platform.py](../../vllm_ascend/platform.py) | 查 `NPUPlatform`：配置验证、后端、Worker、编译和 V2 扩展上下文入口 |
| 4 | [utils.py](../../vllm_ascend/utils.py) | 查 `adapt_patch`、`register_ascend_customop`，理解补丁与层实现的注册时机 |
| 5 | [patch/platform](../../vllm_ascend/patch/platform/)、[patch/worker](../../vllm_ascend/patch/worker/) | 查包导入清单，再进入对应补丁；确认目标上游接口 |
| 6 | [worker/worker.py](../../vllm_ascend/worker/worker.py) | 跟随 `NPUWorker` 的设备初始化、加载、预算和执行入口 |

原理对应详细设计第 3～4 章。不要把安装入口、模型注册、CustomOp 层替换和设备算子绑定视为同一件事。

### 2.3 一轮执行：从请求状态读到设备元数据

| 文件/目录 | 负责的数据 | 应继续跟踪的对象 |
| --- | --- | --- |
| [worker/model_runner_v1.py](../../vllm_ascend/worker/model_runner_v1.py) | 活跃请求与本轮展平输入、执行模式、输出衔接 | `_prepare_inputs`、Attention Builder、上下文、模型 forward |
| [worker/npu_input_batch.py](../../vllm_ascend/worker/npu_input_batch.py) | 批次请求、token、长度与缓存表 | `NPUInputBatch` 与多缓存组 block table |
| [worker/block_table.py](../../vllm_ascend/worker/block_table.py) | 请求逻辑 block 与设备物理 block 的映射 | slot mapping 生成和后端物理布局 |
| [ascend_forward_context.py](../../vllm_ascend/ascend_forward_context.py) | 本轮 MoE 选择、draft/profile/图模式等信息 | `set_ascend_forward_context`、`_EXTRA_CTX`、通信策略 |
| [worker/device_metadata.py](../../vllm_ascend/worker/device_metadata.py) | 设备侧元数据任务的提交、等待与复用 | producer/consumer、stage、group、ready/release 事件 |
| [worker/v2/model_runner.py](../../vllm_ascend/worker/v2/model_runner.py) | 上游 V2 执行与本地状态管理的衔接 | 继承链、model state、抢占处理和 `finally` 清理 |
| [worker/v2/model_states](../../vllm_ascend/worker/v2/model_states/) | 默认或特定模型本轮状态 | 元数据构建、Engram 状态与 `finish_execution` |

详细设计 6.4～6.5 用两个请求解释 `query_start_loc`、`seq_lens`、`positions` 为什么不同；第 18 章解释这些数据什么时候可以安全读取。

### 2.4 模型算法：以 DeepSeek V4.1 串起用户关心的模块

这张表按模型中的联系组织。它不表示视觉、索引、草稿模型等在每一层都执行；是否执行由模型拓扑和当前请求决定。

| 文件 | 核心职责 | 怎样与其他文件对应 |
| --- | --- | --- |
| [models/__init__.py](../../vllm_ascend/models/__init__.py) | 根据 architecture 注册模型类 | 从这里确认实际加载哪个包装类 |
| [deepseek_v41/vl_model.py](../../vllm_ascend/models/deepseek_v41/vl_model.py) | 多模态接入、图像跨度和视觉 embedding | 连接 `vision.py` 与文本骨干 |
| [deepseek_v41/vision.py](../../vllm_ascend/models/deepseek_v41/vision.py) | patch embedding、ViT、空间合并与 Aligner | 视觉向量进入语言模型的入口 |
| [deepseek_v41/model.py](../../vllm_ascend/models/deepseek_v41/model.py) | 拓扑、Q/KV 投影、层执行、mHC、MoE 和辅助状态 | 先看 `build_layer_plan` 与 Decoder/Model 的 forward，再进入子模块 |
| [deepseek_v41/engram](../../vllm_ascend/models/deepseek_v41/engram/) | token 历史查找、表加载、融合和请求状态 | `model.py` 在配置的层入口使用，异步执行还需追 model-state |
| [deepseek_v41/compressor.py](../../vllm_ascend/models/deepseek_v41/compressor.py) | 原始位置到压缩长程表示、循环残留状态 | 输出进入 DSA 的长程缓存与 Indexer K 更新 |
| [deepseek_v41/indexer.py](../../vllm_ascend/models/deepseek_v41/indexer.py) | Query/Key 准备、候选与 TopK 选择 | DSA 调用并把结果交给稀疏注意力 |
| [attention/dsa_v41.py](../../vllm_ascend/attention/dsa_v41.py) | 当前模型的元数据、缓存层和 SFA/MLA 计算组织 | 看 Builder、CacheLayer、Impl，关联投影、压缩和设备调用 |
| [deepseek_v41/cache_config.py](../../vllm_ascend/models/deepseek_v41/cache_config.py) | 模型缓存组、压缩和 packed 布局约定 | 与 `core/kv_cache_interface.py`、Runner 分配共同阅读 |
| [deepseek_v41/dspark.py](../../vllm_ascend/models/deepseek_v41/dspark.py) | 草稿结构、辅助状态融合、草稿上下文 KV | 与 `spec_decode/dspark_proposer.py`、V2 DSpark speculator 对应 |

MLA 中的“共享 KV 表示”、Compressor 的“沿历史位置压缩”、量化的“每个数值使用较少位数”改变的是不同维度。详细设计第 7～16 章分别给出形状、原理和代码入口。

### 2.5 MoE：目录跨得多，是因为计算包含多个阶段

```text
models/ 中的 MoE 层
  → ops/fused_moe/ 路由、派发、专家计算与归并
  → ascend_forward_context.py / distributed/ 确定通信路径与参与组
  → quantization/methods/ 解释专家权重、激活和 scale
  → torch_npu / CANN MegaMoE / csrc/ 执行设备操作
```

| 入口 | 主要问题 |
| --- | --- |
| [ops/fused_moe/fused_moe.py](../../vllm_ascend/ops/fused_moe/fused_moe.py) | `AscendMoERunner` 怎样组织本轮专家计算 |
| [ops/fused_moe/router](../../vllm_ascend/ops/fused_moe/router/) | 专家 ID 与路由权重怎样生成 |
| [ops/fused_moe/moe_comm_method.py](../../vllm_ascend/ops/fused_moe/moe_comm_method.py) | AllGather、AllToAll、MC2 与融合路径怎样准备、执行和收尾 |
| [ops/fused_moe/dataclass](../../vllm_ascend/ops/fused_moe/dataclass/) | 派发结果、共享输入和阶段事件怎样传递 |
| [ops/fused_moe/mega_moe.py](../../vllm_ascend/ops/fused_moe/mega_moe.py) | 本地融合适配怎样验证算子 ABI 与共享专家布局 |
| [eplb](../../vllm_ascend/eplb/) | 专家负载统计与映射调整怎样接入 |

`csrc/moe/` 还包含 mHC 等算子，不能根据父目录名断定每个子目录都实现路由专家。原理和当前融合限制见详细设计第 14 章。

### 2.6 KV、分布式与图：先确认管理对象

| 代码位置 | 管理对象 | 要与谁核对 |
| --- | --- | --- |
| [core/kv_cache_interface.py](../../vllm_ascend/core/kv_cache_interface.py) | 缓存的逻辑/物理描述 | 模型 cache config、Runner allocate/reshape |
| [core/kv_cache_placement.py](../../vllm_ascend/core/kv_cache_placement.py)、[worker/v2/kvpp.py](../../vllm_ascend/worker/v2/kvpp.py) | KVPP 放置计划与执行前取得缓存 | owner rank、预取、消费完成和 scratch 复用 |
| [distributed/parallel_state.py](../../vllm_ascend/distributed/parallel_state.py) | TP、EP 等通信组 | 参与者、rank、调用顺序和模型切分 |
| [distributed/kv_transfer](../../vllm_ascend/distributed/kv_transfer/) | PD、外部缓存池、Offload 的传输 | Scheduler/Worker 两端元数据和层级完成事件 |
| [compilation](../../vllm_ascend/compilation/)、[worker/v2/aclgraph](../../vllm_ascend/worker/v2/aclgraph_utils.py) | 编译、捕获和重放 | 固定地址、padding 容量、任务参数更新与实际有效长度 |
| [model_executor/offloader](../../vllm_ascend/model_executor/offloader/) | 权重预取 | 参数使用时机，不等同于 Attention KV Offload |
| [ec_manager](../../vllm_ascend/ec_manager/) | 多模态编码器结果 | 命中/驱逐语义，不等同于 prefix-cache |

### 2.7 从 Python 一直读到 csrc

先定位 Python 调用方，再确认操作来源。对本仓编译的 Ascend C 算子，常见阅读路径如下。

```text
模型或 ops/ 封装
  → torch.ops._C_ascend.<操作名>
  → csrc/torch_binding.cpp 的 schema / 实现绑定
  → ACLNN/CANN 调用
  → 对应算子的 op_host/ 与 op_kernel/
```

`op_host/` 负责主机侧描述、检查和 tiling；`op_kernel/` 负责设备计算；`CMakeLists.txt` 和构建脚本决定打包内容。Triton、torch_npu 和外部算子包各有入口，不必都经过上述文件。详细设计 21.3～21.4 区分模型层替换、Python 操作注册和设备绑定。

### 2.8 tests 与工程目录怎样辅助理解

`tests/ut/` 适合了解参数约定、边界和错误情况，`tests/e2e/` 用于设备集成、精度和部署场景。先按功能找对应生产代码，再读测试中的输入与预期；测试文件存在不代表本环境已执行通过。

`examples/` 展示配置组合，`benchmarks/` 测量性能，`tools/` 提供调试与检查，`.github/` 管理 CI 与依赖版本，`docs/source/` 包含项目已有的设计文档。初学者可以先读已有的 [输入准备设计](../../docs/source/developer_guide/Design_Documents/ModelRunner_prepare_inputs.md) 和 [补丁设计](../../docs/source/developer_guide/Design_Documents/patch.md)，再和当前实现对照。

本次目录导读参考 DeepWiki 的系统分层，并补充本地 V4.1、Engram、KVPP 和 MegaMoE 的实际关系。页面版本及阅读入口见 [DeepWiki 对照记录](deepwiki-reading-notes.md)。

## 3. 按真实目录树定位分册

下面显示仓库根目录与下一层目录；更深层目录及直接文件在各分册中完整列出。此树由本次收录的实际路径生成。

```text
vllm-ascend/
  .agents/
    skills/
  .claude/
  .gemini/
  .github/
    ISSUE_TEMPLATE/
    actions/
    workflows/
  benchmarks/
    scripts/
    tests/
  cmake/
  csrc/
    aclnn_torch_adapter/
    attention/
    cmake/
    common/
    gmm/
    kernels/
    mc2/
    mla_preprocess/
    moe/
    scripts/
    third_party/
    utils/
  docs/
    hooks/
    overrides/
    source/
    vLLM-ascend-learning/
  examples/
    chat_templates/
    disaggregated_encoder/
    disaggregated_prefill_v1/
    dynamic_bucket_load_balancer/
    epd_disaggregated/
    eplb/
    external_online_dp/
    offline_DSD_k_tuner/
    quantization/
    rfork/
    rl/
  tests/
    e2e/
    ut/
  tools/
    ai_qos/
    bisect/
    docs_codegen/
    trace/
  vllm_ascend/
    _310p/
    _cann_ops_custom/
    attention/
    compilation/
    core/
    device/
    device_allocator/
    distributed/
    ec_manager/
    eplb/
    lora/
    model_executor/
    model_loader/
    models/
    observability/
    ops/
    patch/
    profiler/
    quantization/
    sample/
    spec_decode/
    tokenization/
    trace/
    weight_switch/
    worker/
    xlite/
```

## 4. 全部分册入口

每个分册继续按真实目录分节：先解释子目录，再逐项列出该层文件。请优先阅读 Python 主包与关注的模型，再进入底层算子和对应测试。

### 仓库根目录与协作配置

| 分册 | 条目数 | 目录职责 |
| --- | ---: | --- |
| [.agents/](code-directory-reference/agents.md) | 20 | 开发辅助技能与操作说明，不参与模型前向执行。 |
| [.claude/](code-directory-reference/claude.md) | 1 | 开发助手相关配置或说明，不属于推理运行代码。 |
| [.gemini/](code-directory-reference/gemini.md) | 2 | 开发助手相关配置或说明，不属于推理运行代码。 |
| [.github/](code-directory-reference/github.md) | 119 | GitHub 协作与自动化：CI、工作流、Issue/PR 模板和版本参考。 |
| [根目录文件](code-directory-reference/root.md) | 39 | 仓库根目录：安装构建、容器、开发规范、依赖与项目介绍。 |

### Ascend Python 主包

| 分册 | 条目数 | 目录职责 |
| --- | ---: | --- |
| [vllm_ascend/](code-directory-reference/vllm_ascend.md) | 12 | Ascend 插件 Python 主包：将上游 vLLM 的引擎和模型接口接到 NPU 实现。 |
| [vllm_ascend/_310p/](code-directory-reference/vllm_ascend--_310p.md) | 64 | Ascend 310P 专用执行、输入、算子、采样及量化适配。 |
| [vllm_ascend/_cann_ops_custom/](code-directory-reference/vllm_ascend--_cann_ops_custom.md) | 1 | 自定义 CANN 算子安装/打包位置的占位目录。 |
| [vllm_ascend/attention/](code-directory-reference/vllm_ascend--attention.md) | 24 | Attention 后端、元数据构建、KV 访问和稀疏注意力执行。 |
| [vllm_ascend/compilation/](code-directory-reference/vllm_ascend--compilation.md) | 15 | NPU 图捕获、重放、运行参数更新与编译融合接入。 |
| [vllm_ascend/core/](code-directory-reference/vllm_ascend--core.md) | 10 | 调度器扩展、KV 缓存规格、缓存管理和 KVPP 物理放置。 |
| [vllm_ascend/device/](code-directory-reference/vllm_ascend--device.md) | 7 | 设备识别、硬件能力、策略及设备算子接口。 |
| [vllm_ascend/device_allocator/](code-directory-reference/vllm_ascend--device_allocator.md) | 3 | 设备内存分配及 sleep/wakeup 相关资源管理。 |
| [vllm_ascend/distributed/](code-directory-reference/vllm_ascend--distributed.md) | 4 | 并行通信组、设备通信、专家负载策略以及 KV/权重传输。 |
| [vllm_ascend/distributed/device_communicators/](code-directory-reference/vllm_ascend--distributed--device_communicators.md) | 4 | NPU 设备集合通信及相关通信后端适配。 |
| [vllm_ascend/distributed/ec_transfer/](code-directory-reference/vllm_ascend--distributed--ec_transfer.md) | 2 | 传输 子模块；所属范围：并行通信组、设备通信、专家负载策略以及 KV/权重传输。 |
| [vllm_ascend/distributed/eplb/](code-directory-reference/vllm_ascend--distributed--eplb.md) | 9 | 专家负载均衡 子模块；所属范围：并行通信组、设备通信、专家负载策略以及 KV/权重传输。 |
| [vllm_ascend/distributed/kv_transfer/](code-directory-reference/vllm_ascend--distributed--kv_transfer.md) | 69 | KV 连接器注册、PD 传输、共享缓存池和稀疏缓存迁移。 |
| [vllm_ascend/distributed/weight_transfer/](code-directory-reference/vllm_ascend--distributed--weight_transfer.md) | 4 | 在线权重更新使用的 HCCL/NPU IPC 传输实现。 |
| [vllm_ascend/ec_manager/](code-directory-reference/vllm_ascend--ec_manager.md) | 2 | 多模态编码器结果缓存策略，包括重算成本与访问得分。 |
| [vllm_ascend/eplb/](code-directory-reference/vllm_ascend--eplb.md) | 15 | 专家并行负载均衡：负载收集、专家映射、策略与权重更新适配。 |
| [vllm_ascend/lora/](code-directory-reference/vllm_ascend--lora.md) | 6 | LoRA 在 NPU 上的参数、算子、MoE 和 Punica 接入。 |
| [vllm_ascend/model_executor/](code-directory-reference/vllm_ascend--model_executor.md) | 12 | 模型执行辅助和算子预热；完整执行调度还在 Worker/Runner。 |
| [vllm_ascend/model_loader/](code-directory-reference/vllm_ascend--model_loader.md) | 23 | 可选加载器与权重载入扩展。 |
| [vllm_ascend/models/](code-directory-reference/vllm_ascend--models.md) | 11 | 模型注册、模型专有层结构、权重映射与草稿模型实现。 |
| [vllm_ascend/models/common/](code-directory-reference/vllm_ascend--models--common.md) | 3 | 多个模型复用的执行辅助与序列并行操作。 |
| [vllm_ascend/models/deepseek_v4/](code-directory-reference/vllm_ascend--models--deepseek_v4.md) | 9 | DeepSeek V4 骨干、视觉、压缩、MTP 和 DSpark 实现。 |
| [vllm_ascend/models/deepseek_v41/](code-directory-reference/vllm_ascend--models--deepseek_v41.md) | 17 | DeepSeek V4.1 骨干、层共享拓扑、视觉、稀疏选择及缓存组织。 |
| [vllm_ascend/models/glm5next/](code-directory-reference/vllm_ascend--models--glm5next.md) | 17 | GLM5Next 模型结构和专有计算适配。 |
| [vllm_ascend/models/layer/](code-directory-reference/vllm_ascend--models--layer.md) | 3 | layer 子模块；所属范围：模型注册、模型专有层结构、权重映射与草稿模型实现。 |
| [vllm_ascend/models/minimax_m3/](code-directory-reference/vllm_ascend--models--minimax_m3.md) | 8 | MiniMax M3 模型及多模态相关适配。 |
| [vllm_ascend/observability/](code-directory-reference/vllm_ascend--observability.md) | 5 | 运行观测的 provider、handler 与指标接入。 |
| [vllm_ascend/ops/](code-directory-reference/vllm_ascend--ops.md) | 26 | 模型层与设备 kernel 之间的 Python 算子封装、注册和执行编排。 |
| [vllm_ascend/ops/fused_moe/](code-directory-reference/vllm_ascend--ops--fused_moe.md) | 25 | MoE 路由、token 分发/合并、专家 FFN、共享专家和通信执行方法。 |
| [vllm_ascend/ops/triton/](code-directory-reference/vllm_ascend--ops--triton.md) | 153 | 使用 Triton 编写的设备 kernel 与其调用封装。 |
| [vllm_ascend/patch/](code-directory-reference/vllm_ascend--patch.md) | 71 | 在既有上游接口缺少所需扩展点时安装兼容补丁。 |
| [vllm_ascend/profiler/](code-directory-reference/vllm_ascend--profiler.md) | 2 | 设备性能分析器与运行记录接入。 |
| [vllm_ascend/quantization/](code-directory-reference/vllm_ascend--quantization.md) | 40 | 量化配置解析、层适配、权重后处理与计算方案。 |
| [vllm_ascend/sample/](code-directory-reference/vllm_ascend--sample.md) | 5 | Logits 处理、TopK/TopP、采样和投机验证恢复。 |
| [vllm_ascend/spec_decode/](code-directory-reference/vllm_ascend--spec_decode.md) | 17 | 草稿生成器、候选组织及投机解码辅助。 |
| [vllm_ascend/tokenization/](code-directory-reference/vllm_ascend--tokenization.md) | 2 | 可选的长文本并行 tokenization。 |
| [vllm_ascend/trace/](code-directory-reference/vllm_ascend--trace.md) | 6 | 请求时间线标记、日志和 tracing 包装。 |
| [vllm_ascend/weight_switch/](code-directory-reference/vllm_ascend--weight_switch.md) | 3 | Linear/输出投影完整权重切换和并行域相关生命周期。 |
| [vllm_ascend/worker/](code-directory-reference/vllm_ascend--worker.md) | 10 | Worker 生命周期、V1/V2 Runner、输入组批、缓存分配与设备元数据。 |
| [vllm_ascend/worker/v2/](code-directory-reference/vllm_ascend--worker--v2.md) | 43 | 基于上游 V2 Runner 的 NPU 执行、状态、并行传输和图管理。 |
| [vllm_ascend/xlite/](code-directory-reference/vllm_ascend--xlite.md) | 5 | Xlite 专用 Worker、模型包装及执行适配。 |

### 自定义算子与构建

| 分册 | 条目数 | 目录职责 |
| --- | ---: | --- |
| [cmake/](code-directory-reference/cmake.md) | 1 | 仓库顶层 CMake 构建辅助。 |
| [csrc/](code-directory-reference/csrc.md) | 10 | 自定义设备算子、PyTorch/CANN 绑定及算子构建工具链。 |
| [csrc/aclnn_torch_adapter/](code-directory-reference/csrc--aclnn_torch_adapter.md) | 5 | PyTorch 张量与 ACLNN 算子调用之间的适配。 |
| [csrc/attention/](code-directory-reference/csrc--attention.md) | 862 | 注意力、索引、压缩、缓存写入与相关状态算子。 |
| [csrc/cmake/](code-directory-reference/csrc--cmake.md) | 73 | 算子工程的构建模块、脚本和外部依赖配置。 |
| [csrc/common/](code-directory-reference/csrc--common.md) | 124 | 各算子复用的类型、宏、注册、主机辅助与基础代码。 |
| [csrc/gmm/](code-directory-reference/csrc--gmm.md) | 82 | 分组矩阵乘及激活/量化融合算子。 |
| [csrc/kernels/](code-directory-reference/csrc--kernels.md) | 7 | 独立设备 kernel 和辅助实现。 |
| [csrc/mc2/](code-directory-reference/csrc--mc2.md) | 141 | 通信与专家计算融合的 Dispatch/FFN/Combine 算子。 |
| [csrc/mla_preprocess/](code-directory-reference/csrc--mla_preprocess.md) | 33 | MLA 前处理的主机与设备实现。 |
| [csrc/moe/](code-directory-reference/csrc--moe.md) | 344 | 专家路由/FFN 及历史归在此处的 Norm、mHC、状态和缓存算子。 |
| [csrc/scripts/](code-directory-reference/csrc--scripts.md) | 69 | 算子代码生成、编译产物处理和安装打包脚本。 |
| [csrc/third_party/](code-directory-reference/csrc--third_party.md) | 1 | 通过子模块接入的第三方算子开发依赖。 |
| [csrc/utils/](code-directory-reference/csrc--utils.md) | 20 | 设备/主机辅助函数、接口和工具实现。 |

### 测试

| 分册 | 条目数 | 目录职责 |
| --- | ---: | --- |
| [tests/](code-directory-reference/tests.md) | 5 | 单元、系统、集成、性能和测试基础设施；不是服务运行入口。 |
| [tests/e2e/](code-directory-reference/tests--e2e.md) | 7 | 模型、部署和多设备场景的端到端验证。 |
| [tests/e2e/cases/](code-directory-reference/tests--e2e--cases.md) | 214 | 按模型或功能组织的测试场景。 |
| [tests/e2e/common/](code-directory-reference/tests--e2e--common.md) | 23 | 端到端场景共用的部署与断言辅助。 |
| [tests/e2e/doctests/](code-directory-reference/tests--e2e--doctests.md) | 5 | 端到端 doctests 场景、运行脚本和配置。 |
| [tests/e2e/models/](code-directory-reference/tests--e2e--models.md) | 28 | 模型级测试配置及辅助定义。 |
| [tests/e2e/nightly/](code-directory-reference/tests--e2e--nightly.md) | 144 | 夜间模型、精度、稳定性与性能场景。 |
| [tests/e2e/prompts/](code-directory-reference/tests--e2e--prompts.md) | 3 | 端到端 prompts 场景、运行脚本和配置。 |
| [tests/e2e/pull_request/](code-directory-reference/tests--e2e--pull_request.md) | 192 | 用于 PR 场景的分设备数量测试和运行脚本。 |
| [tests/e2e/vllm_interface/](code-directory-reference/tests--e2e--vllm_interface.md) | 13 | 端到端 vllm_interface 场景、运行脚本和配置。 |
| [tests/ut/](code-directory-reference/tests--ut.md) | 20 | 围绕函数、组件或接口契约的单元/回归测试。 |
| [tests/ut/_310p/](code-directory-reference/tests--ut--_310p.md) | 33 | 验证 _310p 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/_fake_weight/](code-directory-reference/tests--ut--_fake_weight.md) | 2 | 验证 _fake_weight 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/_tools/](code-directory-reference/tests--ut--_tools.md) | 9 | 验证 _tools 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/attention/](code-directory-reference/tests--ut--attention.md) | 39 | 验证 attention 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/compilation/](code-directory-reference/tests--ut--compilation.md) | 5 | 验证 compilation 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/core/](code-directory-reference/tests--ut--core.md) | 14 | 验证 core 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/device/](code-directory-reference/tests--ut--device.md) | 4 | 验证 device 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/device_allocator/](code-directory-reference/tests--ut--device_allocator.md) | 5 | 验证 device_allocator 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/distributed/](code-directory-reference/tests--ut--distributed.md) | 50 | 验证 distributed 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/eplb/](code-directory-reference/tests--ut--eplb.md) | 10 | 验证 eplb 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/helpers/](code-directory-reference/tests--ut--helpers.md) | 1 | 验证 helpers 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/kv_offload/](code-directory-reference/tests--ut--kv_offload.md) | 43 | 验证 kv_offload 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/kv_transfer/](code-directory-reference/tests--ut--kv_transfer.md) | 4 | 验证 kv_transfer 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/lora/](code-directory-reference/tests--ut--lora.md) | 6 | 验证 lora 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/model_executor/](code-directory-reference/tests--ut--model_executor.md) | 13 | 验证 model_executor 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/model_loader/](code-directory-reference/tests--ut--model_loader.md) | 19 | 验证 model_loader 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/models/](code-directory-reference/tests--ut--models.md) | 47 | 验证 models 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/observability/](code-directory-reference/tests--ut--observability.md) | 1 | 验证 observability 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/ops/](code-directory-reference/tests--ut--ops.md) | 43 | 验证 ops 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/patch/](code-directory-reference/tests--ut--patch.md) | 35 | 验证 patch 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/profiler/](code-directory-reference/tests--ut--profiler.md) | 2 | 验证 profiler 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/proxy/](code-directory-reference/tests--ut--proxy.md) | 3 | 验证 proxy 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/quantization/](code-directory-reference/tests--ut--quantization.md) | 42 | 验证 quantization 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/sample/](code-directory-reference/tests--ut--sample.md) | 5 | 验证 sample 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/simple_kv_offload/](code-directory-reference/tests--ut--simple_kv_offload.md) | 2 | 验证 simple_kv_offload 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/spec_decode/](code-directory-reference/tests--ut--spec_decode.md) | 25 | 验证 spec_decode 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/tokenization/](code-directory-reference/tests--ut--tokenization.md) | 1 | 验证 tokenization 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/tools/](code-directory-reference/tests--ut--tools.md) | 13 | 验证 tools 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/trace/](code-directory-reference/tests--ut--trace.md) | 1 | 验证 trace 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/worker/](code-directory-reference/tests--ut--worker.md) | 50 | 验证 worker 对应组件的单元/回归行为；子目录按生产模块组织。 |
| [tests/ut/xlite/](code-directory-reference/tests--ut--xlite.md) | 2 | 验证 xlite 对应组件的单元/回归行为；子目录按生产模块组织。 |

### 文档

| 分册 | 条目数 | 目录职责 |
| --- | ---: | --- |
| [docs/](code-directory-reference/docs.md) | 3 | 项目文档、站点构建、主题覆盖与当前学习资料。 |
| [docs/hooks/](code-directory-reference/docs--hooks.md) | 3 | 父主题的 hooks 文档、模板或站点资源。 |
| [docs/overrides/](code-directory-reference/docs--overrides.md) | 2 | 父主题的 overrides 文档、模板或站点资源。 |
| [docs/source/](code-directory-reference/docs--source.md) | 3 | 用户与开发文档正文、图片、模板及翻译资源。 |
| [docs/source/_templates/](code-directory-reference/docs--source--_templates.md) | 8 | 父主题的 _templates 文档、模板或站点资源。 |
| [docs/source/assets/](code-directory-reference/docs--source--assets.md) | 26 | 父主题的示意图、截图和文档展示资源。 |
| [docs/source/community/](code-directory-reference/docs--source--community.md) | 10 | 社区协作、发布与用户实践材料。 |
| [docs/source/developer_guide/](code-directory-reference/docs--source--developer_guide.md) | 38 | 开发架构、贡献规范、测试评估和调优排错说明。 |
| [docs/source/getting_started/](code-directory-reference/docs--source--getting_started.md) | 24 | 环境安装与首次运行指引。 |
| [docs/source/javascripts/](code-directory-reference/docs--source--javascripts.md) | 2 | 父主题的 javascripts 文档、模板或站点资源。 |
| [docs/source/locale/](code-directory-reference/docs--source--locale.md) | 206 | 文档翻译资源，不是运行期模型逻辑。 |
| [docs/source/logos/](code-directory-reference/docs--source--logos.md) | 2 | 父主题的 logos 文档、模板或站点资源。 |
| [docs/source/stylesheets/](code-directory-reference/docs--source--stylesheets.md) | 1 | 父主题的 stylesheets 文档、模板或站点资源。 |
| [docs/source/tutorials/](code-directory-reference/docs--source--tutorials.md) | 62 | 模型与功能使用教程。 |
| [docs/source/user_guide/](code-directory-reference/docs--source--user_guide.md) | 56 | 配置、部署、功能及支持范围说明。 |
| [docs/vLLM-ascend-learning/](code-directory-reference/docs--vLLM-ascend-learning.md) | 7 | 本次源码学习、详细设计、PR 分析与适配说明。 |

### 示例、性能与开发工具

| 分册 | 条目数 | 目录职责 |
| --- | ---: | --- |
| [benchmarks/](code-directory-reference/benchmarks.md) | 10 | 性能测量脚本、输入配置和报告模板。 |
| [examples/](code-directory-reference/examples.md) | 52 | 离线推理、分布式部署、权重更新和量化等参考使用示例。 |
| [tools/](code-directory-reference/tools.md) | 54 | 开发检查、调试、请求发送、文档生成与性能工具。 |

## 5. 按问题找文件

| 我想了解…… | 先看目录/文件 | 再跟进 |
| --- | --- | --- |
| 插件怎样接入引擎 | `setup.py`、`vllm_ascend/__init__.py`、`platform.py` | Worker 选择、后端和配置验证 |
| 一次请求怎样执行 | `worker/worker.py`、`worker/model_runner_v1.py`、`worker/v2/` | 输入批次、缓存、模型 forward、采样 |
| DeepSeek V4.1 的整体结构 | `models/deepseek_v41/model.py`、`vl_model.py` | `attention/dsa_v41.py` 与该模型子模块 |
| ViT、Engram、Indexer、Compressor | `models/deepseek_v41/` 对应文件 | `ops/triton/`、缓存配置与元数据 |
| MoE 路由、专家和融合 | `ops/fused_moe/` | `quantization/methods/`、`csrc/moe/`、`csrc/mc2/` |
| KV 放在哪里、怎样传 | `core/kv_cache_interface.py`、Runner、`distributed/kv_transfer/` | 模型 cache config、KVPP、Offload |
| DSpark 候选与验证 | `spec_decode/`、`worker/v2/spec_decode/`、模型的 `dspark.py` | `sample/rejection_sampler.py`、草稿上下文缓存 |
| 一个 torch.ops 调用的实现 | `csrc/torch_binding.cpp` | 对应算子的 `op_host/`、`op_kernel/` |
| 查行为例子或边界 | `tests/ut/` 对应生产模块目录 | 对应端到端、精度和设备测试 |

## 6. 使用与更新边界

文件功能说明不代表该路径始终启用。实际入口取决于 architecture、上游版本、硬件能力、量化、并行与图配置。阅读 `models/` 时先核对模型注册表；阅读算子时先确认 Python 调用方及 ABI。

新增或移动源码后，需要重新生成/补充该索引；本页数字是上述提交与当前本地附加资料的快照。索引中的外部 Git 子模块只描述引用，不能作为其内部实现的说明。
