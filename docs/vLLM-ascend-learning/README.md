# vLLM Ascend 学习与设计文档

??????? `integrate/ds-v41-flash-2db854`??????????? `2db8549c5`?v0.30 ??????????? [Flash ????](deepseek-v41-flash-vllm-cann-adaptation.md)??????????????

这套文档面向第一次接触推理框架的读者，按“先理解系统怎样执行，再理解模型怎样计算，最后查阅具体文件”的顺序组织。2026-10-15 已根据仓库 README 链接的 DeepWiki 重新整理，并对照本地源码补充和纠正实现细节。

本地版本：`integrate/dsnext-megamoe-vllm-cann` / `203fbd7d43268cbca4ecb6b2a8a8aa4f43e57839`。DeepWiki 各页面有自己的索引版本，区别见 [阅读记录与源码对照](deepwiki-reading-notes.md)。本文档集依据静态分析，未运行 NPU 推理或性能实验。

## 建议的阅读顺序

| 顺序 | 文档与章节 | 读完能回答什么 |
| --- | --- | --- |
| 1 | [仓库介绍与模块地图](repository-module-overview.md) | vLLM 与插件各负责什么，用户关心的模块处在哪一层 |
| 2 | [详细设计](detailed-design.md) 第 2～6 章 | token、张量、Prefill/Decode、启动、权重、组批和 Forward Context 是什么 |
| 3 | 同文第 7～14 章 | Attention、ViT、Compressor、Indexer、SFA、Engram、mHC、MoE 怎样计算 |
| 4 | 同文第 15～21 章 | KV、DSpark、多卡、异步流、图执行和采样怎样支撑完整请求 |
| 5 | [目录结构与关键文件导读](code-directory-guide.md) | 一项功能应沿哪些文件、对象和调用关系阅读 |
| 6 | 导读中的逐文件分册 | 某个具体文件负责什么，有哪些定义和源码依据 |

目录导读提供“为何在这里、上下游是谁”的解释；逐文件表用于查找。遇到陌生术语，可回看详细设计第 24 章；想检验理解，可做第 23 章的纸面练习。

## 来源与变更资料

当前原始 `rfc/vllm_cann` 分支（`7e37e29a7`）的 MegaMoE 专项分析见下面第一项，其版本范围与其他历史适配资料分开标注。

| 文档 | 用途 |
| --- | --- |
| [原始 rfc/vllm_cann：MegaMoE 接入架构与执行分析](rfc-vllm-cann-megamoe-architecture.md) | 支持矩阵、共享专家边界、策略选择、量化权重、token/容量、多 rank 和算子 buffer 的完整调用分析 |
| [DeepWiki 阅读记录与当前源码对照](deepwiki-reading-notes.md) | 阅读主题、页面版本、当前代码入口与需要限定的结论 |
| [PR18：A5 MegaMoE 方案与代码逐项学习](pr18-megamoe-design-and-code-walkthrough.md) | 原始17文件、关键代码、完整调用链及当前版本对照；附完整原始patch |
| [vllm_cann 适配说明](dsnext-vllm-cann-adaptation.md) | 此前三个 PR 如何适配当前基线 |
| [其他 MoE PR 分析](dsnext-other-moe-pr-analysis.md) | 来源仓其他 MoE 改动及当前覆盖情况 |

这套学习资料描述当前本地分支。切换代码版本、上游依赖、硬件或量化方案后，需重新核对注册入口、配置验证、张量布局和异步资源生命周期。
