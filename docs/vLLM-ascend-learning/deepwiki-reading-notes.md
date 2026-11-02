# DeepWiki 阅读记录与当前源码对照

## 1. 本次阅读依据

阅读日期：2026-10-15。入口来自仓库 [README 的 DeepWiki 章节](../../README.md)。外部入口为 [vLLM Ascend DeepWiki](https://deepwiki.com/vllm-project/vllm-ascend)。

本次围绕系统架构、平台接入、Worker/Runner、Forward Context、Attention、KV、分布式、图执行、量化、算子注册和目录结构阅读相关页面。DeepWiki 用来组织问题和发现源码入口；本目录关于当前行为的说明以本地源码为依据。

当前源码为 `integrate/dsnext-megamoe-vllm-cann` / `203fbd7d43268cbca4ecb6b2a8a8aa4f43e57839`，基线为 `rfc/vllm_cann` 的 `0040652a66d31a2b277588ac90d7d37e92757bc5`。本文是静态阅读记录，没有执行模型推理。

### 1.1 DeepWiki 页面具有各自的版本

| 读取页面 | 页面显示的索引日期 | 页面显示的提交标记 | 阅读时的处理 |
| --- | --- | --- | --- |
| [Overview](https://deepwiki.com/vllm-project/vllm-ascend) | 2026-10-13 | `d280bb` | 用于理解仓库总体定位 |
| [Code Organization and Structure](https://deepwiki.com/vllm-project/vllm-ascend/8.2-code-organization-and-structure) | 2026-09-13 | `660c45` | 目录名称逐一核对本地路径 |
| [Forward Context and Metadata Flow](https://deepwiki.com/vllm-project/vllm-ascend/3.4-forward-context-and-metadata-flow) | 2026-10-02 | `11682d` | V1/V2、字段和状态生命周期以本地实现为准 |

这些标记记录的是所读取页面的显示信息，不能视为本地代码的版本，也不能代表整个站点使用同一个快照。后续访问时页面可能更新。DeepWiki 引用的源码行号同样不能直接套用到本地文件。

## 2. 阅读主题与代码对应

按下表顺序阅读，可以先建立执行框架，再进入具体模型。每行给出本地核对入口，继续查找目录中的类、方法及其调用者即可。

| DeepWiki 主题 | 读完应理解的问题 | 当前源码核对入口 | 本目录落点 |
| --- | --- | --- | --- |
| [系统架构](https://deepwiki.com/vllm-project/vllm-ascend/1.3-system-architecture-overview) | 插件怎样承接上游引擎 | [setup.py](../../setup.py)、[platform.py](../../vllm_ascend/platform.py) | 详细设计第 3～4 章 |
| [平台集成](https://deepwiki.com/vllm-project/vllm-ascend/3.1-platform-layer-and-npuplatform) | 平台发现、注册和配置发生在哪里 | [包初始化](../../vllm_ascend/__init__.py)、[平台补丁](../../vllm_ascend/patch/platform/) | 详细设计 3.2、3.4 |
| [Worker 与设备](https://deepwiki.com/vllm-project/vllm-ascend/3.2-worker-system-and-initialization) | 谁管理设备、加载和内存预算 | [worker.py](../../vllm_ascend/worker/worker.py)、[device](../../vllm_ascend/device/) | 详细设计 4.1～4.4 |
| [Runner 与执行](https://deepwiki.com/vllm-project/vllm-ascend/3.3-model-runner-and-execution-pipeline) | 调度结果怎样变成模型输入 | [V1 Runner](../../vllm_ascend/worker/model_runner_v1.py)、[V2 Runner](../../vllm_ascend/worker/v2/model_runner.py) | 详细设计第 6、20 章 |
| [Forward Context](https://deepwiki.com/vllm-project/vllm-ascend/3.4-forward-context-and-metadata-flow) | 深层算子怎样取得当前执行状态 | [ascend_forward_context.py](../../vllm_ascend/ascend_forward_context.py)、[device_metadata.py](../../vllm_ascend/worker/device_metadata.py) | 详细设计 6.4～6.5 |
| [配置与环境变量](https://deepwiki.com/vllm-project/vllm-ascend/4-configuration-system) | 用户选项怎样经过能力校验 | [ascend_config.py](../../vllm_ascend/ascend_config.py)、[envs.py](../../vllm_ascend/envs.py)、[hardware_profile.py](../../vllm_ascend/device/hardware_profile.py) | 详细设计 4.5、21.1 |
| [Attention 后端](https://deepwiki.com/vllm-project/vllm-ascend/3.6-attention-backends-deep-dive) | Backend、Builder、Impl 怎样配合 | [attention](../../vllm_ascend/attention/)、[dsa_v41.py](../../vllm_ascend/attention/dsa_v41.py) | 详细设计 7.5 及第 9～11 章 |
| [内存与 KV](https://deepwiki.com/vllm-project/vllm-ascend/3.7-memory-management-and-kv-cache) | 逻辑页、布局与物理分配怎样连接 | [kv_cache_interface.py](../../vllm_ascend/core/kv_cache_interface.py)、[attn_utils.py](../../vllm_ascend/worker/v2/attn_utils.py) | 详细设计第 15 章 |
| [分布式概览](https://deepwiki.com/vllm-project/vllm-ascend/7-distributed-and-multi-node-execution) | TP、EP、DP、PP 与上下文并行切分什么 | [distributed](../../vllm_ascend/distributed/)、[context_parallel](../../vllm_ascend/attention/context_parallel/) | 详细设计第 17 章 |
| [Prefill/Decode 分离](https://deepwiki.com/vllm-project/vllm-ascend/7.4-disaggregated-prefill-decode-architecture) | 分离部署时如何交接 KV | [kv_transfer](../../vllm_ascend/distributed/kv_transfer/) | 详细设计 15.8、17.4 |
| [MoE 与专家并行](https://deepwiki.com/vllm-project/vllm-ascend/6.2-fused-moe-operations) | token 如何派发并归并 | [fused_moe](../../vllm_ascend/ops/fused_moe/)、[eplb](../../vllm_ascend/eplb/) | 详细设计第 14 章 |
| [ACL Graph](https://deepwiki.com/vllm-project/vllm-ascend/5.1-aclgraph-capture-and-replay) | 捕获、重放、更新怎样减少提交开销 | [compilation](../../vllm_ascend/compilation/)、[V2 aclgraph](../../vllm_ascend/worker/v2/aclgraph_utils.py) | 详细设计第 18 章 |
| [量化](https://deepwiki.com/vllm-project/vllm-ascend/6.3-quantization-system) | 权重格式、scale 和算子怎样匹配 | [quantization](../../vllm_ascend/quantization/) | 详细设计 5.6 |
| [算子注册](https://deepwiki.com/vllm-project/vllm-ascend/6.1-custom-operations-registration) | Python 层替换与 torch.ops 注册各做什么 | [utils.py](../../vllm_ascend/utils.py)、[register_custom_ops.py](../../vllm_ascend/ops/register_custom_ops.py)、[torch_binding.cpp](../../csrc/torch_binding.cpp) | 详细设计 21.3～21.4 |
| [目录结构](https://deepwiki.com/vllm-project/vllm-ascend/8.2-code-organization-and-structure) | 各目录如何服务上述执行链 | [Python 主包](../../vllm_ascend/)、[csrc](../../csrc/)、[tests](../../tests/) | 按目录结构导读与逐文件分册 |

## 3. 对照时需要纠正或限定的地方

| 阅读时容易产生的理解 | 本地核对结果 | 文档处理方式 |
| --- | --- | --- |
| 平台补丁只会在主进程执行一次 | `_ensure_global_patch()` 考虑了 EngineCore 子进程；平台预注册也调用 `adapt_patch(is_global_patch=True)` | 按进程和初始化阶段说明，不把父进程注册视为所有进程已完成 |
| V1/V2 的扩展上下文可以直接互用 | `_ExtraForwardContextProxy` 在 V1 读写对象属性，在 V2 读写 `additional_kwargs` | 解释统一访问入口及其边界 |
| Worker 页面把 `register_ascend_customop()` 概括为 C++ kernel 绑定 | 本地该函数主要调用 `CustomOp.register_oot` 注册层替换；设备 schema/实现另在 C++ 绑定中 | 补充模型层、Python 操作和编译扩展三层注册的区别 |
| 一个 Attention 模块就是一个设备算子 | Backend 描述接口，Builder 生产元数据，Impl 组织计算；内部可能调用多个算子 | 补充三者职责及 V4.1 专用 cache-layer 注册关系 |
| 概览中的 Multi-level、架构页中的 Linear 等 MLA 全称表述可以直接使用 | MLA 在此指 Multi-head Latent Attention；本地 V4.1 的 Q 仍有多个头，共享 KV 表示与查询头数是两件事 | 使用 Multi-head Latent Attention 全称，按当前 Q/KV 投影解释 |
| 量化页面中的示例路径可以直接在本地找到 | 例如本地 `methods/wna16/` 和 `methods/kv_cache/` 是目录，具体方案还需结合 registry 与 adapter | 使用本地真实链接，先追配置到方案再追设备操作 |
| 所有 MLA 都可以套用经典 V2 的展开结构 | 本地 V4.1 使用 `wkv` 生成共享表示，并结合压缩和稀疏索引；执行不等同于旧模型路径 | 先讲通用思想，再画当前代码中的数据变化 |
| KV 管理等于把一张 K/V 张量放进显存 | 当前还涉及 SWA、长程压缩、索引、循环状态、分组、Offload、PD 与 KVPP | 分开说明状态种类、存储布局和部署方式 |
| 文档列出某种硬件就说明所有功能可用 | 本地能力、policy、模型 shape、量化和并行配置共同决定路径 | 避免把支持清单当作所有组合的保证 |
| 整仓模块都由 DeepSeek V4.1 定义 | 框架同时服务多个模型；Engram、mHC 等属于特定模型路径 | 将框架执行与模型算法分层 |

## 4. 本次文档如何重新组织

[学习入口](README.md) 规定阅读顺序；[模块介绍](repository-module-overview.md) 建立总体地图；[详细设计](detailed-design.md) 解释原理、对象、时序和约束；[目录导读](code-directory-guide.md) 先给出关键文件之间的关系，再连接逐文件分册。

原有逐文件分册保留为查询资料。其职责说明包含人工整理的核心入口和从声明、注释提取的辅助说明；它们不能替代对调用关系的阅读。此前三个 PR 的适配记录和其他 MoE PR 分析保留为变更材料，与当前设计说明一起使用。
