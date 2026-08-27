# Kernel 侧 API 文档预研

## 定位

API 预研报告是通用检视子 agent 查阅 API 物理约束的主要来源，覆盖代码中核心 API 的对齐要求、参数限制、配对规则和精度约束。不同芯片代际的 API 约束存在差异（部分代际才支持 RegBase/SIMT/FP8 等路线、Buffer 层级与对齐单位等也可能不同），预研必须先分析目标代际再据此查阅。

file-review 和 pr-review 共用。Phase 0 与 code-summarize **并行**执行。

仅当 code-summarize 判定侧别包含 Kernel 时触发；纯 Tiling 侧跳过。

## 派发

```
Agent({
  subagent_type: "general",
  model: "opus",
  description: "Kernel API 文档预研",
  prompt: "Kernel 侧 API 文档预研

【输入】
- 代码文件：{file_input}（Kernel 侧文件列表）
- 概要输出路径：{api_prestudy_path}

【执行流程】

Step 0 — 分析芯片代际

不同代际 API 约束不同,必须先分析目标芯片代际。从代码的目录结构（如 op_kernel 下的 arch 命名目录）、代码中的架构宏/标记、运行时架构获取调用等线索自行分析代际,使用 /npu-arch skill 核对代际判定与该代际的特有能力（如某些代际才支持 RegBase/SIMT/FP8 等）。
若代际无法确定,标注"代际未确认",后续查阅按保守原则（取约束更严的代际）并标注约束可能不匹配。

Step 1 — 提取代码中使用的 API

1.1 Read 代码文件，提取所有 AscendC:: 命名空间下的函数调用。
1.2 按以下核心 API 清单筛选需要预研的 API（仅保留代码中实际使用的）：

| 类别 | API | 学习重点 |
|------|-----|---------|
| 数据搬运 | DataCopy, DataCopyPad | 32 字节对齐要求、同步机制（EnQue/DeQue 配对）、blockLen 单位（代际差异） |
| 内存管理 | InitBuffer, AllocTensor, FreeTensor, EnQue, DeQue | 配对要求、UB 容量限制、生命周期 |
| 向量计算 | Add, Sub, Mul, Div, Cast | 参数限制（repeatTimes≤255）、RoundMode 正确性 |
| 归约操作 | ReduceSum, ReduceMax | FP32 中间精度保护、累加器 dtype |

1.3 若代码中出现核心清单外的 API（如 Compare, Exp, Sqrt），也一并记录。
1.4 结合 Step 0 分析的代际,若该代际有专属 API 路线（如 RegBase/SIMT 类:RegTensor/MaskReg/asc_vf_call/__simd_vf__/LoadAlign/StoreAlign 等）且代码中实际使用,单独预研其约束。

Step 2 — 按代际查阅官方文档

对 Step 1 提取的每个 API，使用 /ascendc-docs-search skill 查阅官方文档，提取：
- 函数签名（参数类型、返回值）
- 对齐要求（字节数）
- 参数限制（repeatTimes、dtype 约束、shape 约束）
- 配对/同步要求（哪些 API 必须成对使用）
- 精度约束（中间精度、累加器要求）
- RoundMode 选项及默认值
- **代际差异**:该 API 在 Step 0 分析的代际下是否有特定约束或与其他代际的差异（对齐单位、参数语义、支持 dtype、Buffer 层级等）。无差异标注"代际无关";有差异明确列出。使用 /npu-arch skill 核对该代际相对其他代际的关键变化是否影响该 API。

Step 2.5 — 日落 API 比对

2.5.1 运行 `python3 {skill_base}/scripts/clause.get_sunset_api.py` 动态拉取最新 CANN 废弃接口/头文件清单（~2-3 秒）。离线/失败 → 跳过 2.5.2，报告「## 日落 API」写"获取失败，clause-review 需 fallback"。

2.5.2 将 Step 1 提取的代码 API 清单与日落清单比对：
- `aclrt*` / `aclnn*` / `acl.op.*` 符号词法边界匹配（`aclrtGetVersion` 不误匹配替代品 `aclrtGetVersionV2`）
- 头文件/库：比对 `#include` 路径片段（`op_proto/inc`）与链接库名（`libopapi.so`）
- 命中记录：符号、{file}:{line}、替代符号、删除期限（若有）

2.5.3 结果写入报告「## 日落 API」章节（见 Step 3 格式）。

Step 3 — 输出预研报告

将预研结果写入 {api_prestudy_path}，格式如下：

# API 预研报告

## 芯片代际

- 分析代际: {AI 自行从目录结构/代码标记分析得出的代际,或"未确认"}
- 判定依据: {分析过程依据的线索}
- 代际特有能力: {该代际支持的特有 API 路线/能力,或"无"}

## 数据搬运

### DataCopy
- 代际差异: {代际无关 / 该代际下的特定约束}
- 对齐要求：src/dst 地址必须 32 字节对齐
- 同步机制：需配合 EnQue/DeQue 使用
- 参数限制：...
- 代码中的使用位置：{file}:{line}

### DataCopyPad
- ...

## 内存管理

### InitBuffer
- ...

（每个 API 一个子章节,均含「代际差异」字段）

## 代际专属 API（若该代际有专属 API 路线且代码使用）

### {代际专属 API 名}
- ...

## 未匹配 API（代码中使用但不在核心清单中的）

- {API名}: {一行描述}

## 日落 API

- 无命中 → "未检测到日落 API 使用"
- 有命中 → 逐条：{日落符号} ({file}:{line}) → 替代 {替代符号} [删除期限 {deadline}]
- 获取失败 → "日落清单获取失败，clause-review 需 fallback"

禁止生成报告文件以外的输出。"
})
```

## 输出

- 预研报告路径：`./operators/{operator_name}/api_prestudy.md`
- 后续 Phase 1 的 clause-review sub-agent 在检视 API-\*、PERF-\*、PREC-\* 条款时，Read 此文件获取 API 约束上下文（含代际维度），减少重复查阅
