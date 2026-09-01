# 设计文档探测

文件检视、PR 检视、大型 PR 检视共用。派发为 `general` 子 Agent 执行，与 code-summarize / api-prestudy **同级并行**，互不依赖。

合并后设计文档路径由 skill 自行探测（不再依赖调用方传入）。返回 `docs_input`（文件/目录路径，可多个）或空。空表示未检测到设计文档，后续 design-check 不派发。

## 派发

```
Agent({
  subagent_type: "general",
  model: "haiku",
  description: "设计文档探测",
  prompt: "设计文档探测

【输入】
- 代码文件：{file_input}（文件检视：单个或多个路径）
- （PR 模式）完整源码路径：{repo_path}
- 用户已指明的文档路径：{user_docs_hint}（用户在输入中明确给出则传入，否则为空）

【目标】

为待检视的算子代码找到其对应的设计文档（描述算子功能、架构、API、数据流、约束等的 .md/.yaml 文件），供后续设计一致性检查对照。**返回设计文档所在的 docs 目录路径**（而非单个文件），使 design-check 能枚举该目录下所有 .md 文档（含 README、DESIGN.md、aclnn{Op}.md 等）进行设计对照与文档格式检视。

【约束】

- 用户已指明路径（{user_docs_hint} 非空）→ 优先采用，校验有效后返回
- 否则由你自行判断：代码与哪些文档相关。你拥有完全的判断自由——可读代码、grep 线索、浏览目录结构、匹配算子名/接口名/功能语义,用你认为最可靠的方式定位
- 设计文档常见于算子目录的 docs/ 下（DESIGN.md / REQUIREMENTS.md / aclnn{Op}.md 等），但**不局限于此**——算子可能复用同族其他算子的文档、文档可能在上级目录、也可能只有单个 .md 文件
- **返回目录优先**：找到设计文档后，返回其所在的 docs 目录路径（向上回溯到 docs/ 根目录）。仅当文档为孤立单文件（无同级 docs 目录）时返回文件路径
- 跳过过程产物（非设计基准）：LOG.md、*_REVIEW.md、*-report.md（precision/performance 报告）
- 找到 → 返回路径（可多个，逗号分隔）；确实找不到 → 返回空。宁可返回空，不要硬凑无关文档

【输出】

docs_input: {文档路径或目录，可多个逗号分隔；未检测到则为空}
探测依据: {简述你怎么判定这些文档与代码相关，如\"本算子docs/\"/\"关联算子X的docs/，因...\"/\"用户指定\"/\"未检测到\"}

禁止生成报告文件。"
})
```

## 输出

- `docs_input`（路径字符串，可多个逗号分隔）或空字符串
- 后续：file-review / pr-review / pr-large-review 的 Stage1 据此决定是否并行派发 design-check；专项检视子 agent（design-check）据此读取设计文档做 S1-S7 + D8 对照。
