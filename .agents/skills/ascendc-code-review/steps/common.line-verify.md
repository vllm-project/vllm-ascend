# 行号校对（文件检视 + 设计一致性）

## 执行步骤

调用 `{skill_base}/scripts/workflow.line_verify.py` 扫描 yaml 目录，原地修正行号。

**文件检视模式**（无 diff，设计一致性 ❌ 项也走此路径，无 diff 红线）：

```bash
python3 {skill_base}/scripts/workflow.line_verify.py --dir {yaml_output_dir} --repo {repo_path或源码根目录}
```

**脚本行为**：
- 扫描 `{yaml_output_dir}` 下所有 yaml 文件
- 对 clause 类 yaml 的 FAIL/SUSPICIOUS 项：grep 源码定位实际行号，原地更新 `code_snippet.start_line/end_line`
- 对 design 类 yaml 的 `deviations` / `doc_violations`：校对 `code_location` / `violation_location` 行号
- 无 diff 红线（文件检视不做范围过滤，design 类始终不做范围过滤）
- 校对失败的项标记 `line_verified: false`

## 输入

- `yaml_output_dir`：阶段0 router 创建的 yaml 输出目录路径
- `repo_path` 或源码根目录：用于行号定位（文件检视时若 file_input 是绝对路径可不传）

## 输出

- yaml 文件原地更新行号字段
- stdout 打印校对摘要（处理 yaml 数、FAIL/SUSPICIOUS 项数、行号修正数、待确认数）

## 约束

- 本 step 为主 Agent 直接执行，不派发子 Agent
- 行号校对逻辑由脚本统一处理，主 Agent 不手动 grep/read 源码
