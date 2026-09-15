\---

name: phi3.5-moe-instruct-testing

description: 为微软 phi-3.5-moe-instruct 大模型搭建端到端测试体系，包含测试用例设计、Python 脚本改造、目录规范与教程文档。适用于资源受限场景下的模型验证、推理测试与能力评估。

license: proprietary

metadata:

&#x20; author: AI Agent Skill

&#x20; version: 1.0

&#x20; model: microsoft/phi-3.5-moe-instruct

compatibility: requires python 3.10+, PyYAML, requests, Ollama, and access to the local model registry

\---



\# phi-3.5-moe-instruct 技能文档（SKILL.md）



\## 1. 技能目标（skill purpose）

本技能用于完成 \*\*Verify/Support microsoft/phi-3.5-moe-instruct\*\* 任务的全流程交付，包括：

\- 为 moe 架构大模型设计规范的测试用例

\- 改造测试脚本实现配置与代码分离

\- 按项目规范搭建目录结构

\- 编写可复现的教程文档

\- 沉淀可复用的工程化模式



\## 2. 交付物结构（Deliverables）

按官方规范组织文件结构如下：

phi3-5-moe-instruct-testing/

├── SKILL.md          # 本技能文档

├── scripts/          # 测试执行脚本

│   ├── test\_ollama\_phi35.py

│   └── load\_cases.py

├── references/       # 用例配置与文档

│   ├── configs/

│   │   └── phi3.5-moe-instruct-test-cases.yaml

│   └── phi3.5-moe-test-guide.md

└── assets/



\## 3. 技能内容与执行步骤（instructions）



\### 3.1 测试用例设计（Test Case Design）

为 phi-3.5-MoE-instruct 设计 8 类典型能力测试场景，完全对齐用户实际运行的脚本逻辑：



\- 基础常识问答

\- 逻辑推理题

\- 代码生成

\- 数学计算

\- 长文本摘要

\- 多轮对话一致性

\- 指令遵循

\- 中法翻译



用例统一使用 YAML 配置管理，位于：

`references/configs/phi3.5-moe-instruct-test-cases.yaml`



\### 3.2 脚本工程化改造（Script Refactoring）

基于用户现有 Python 脚本，实现：

\- 主脚本 `test\_ollama\_phi35.py` 负责模型请求与统计

\- 独立模块 `load\_cases.py` 负责读取 YAML 用例

\- 路径自适应处理，避免绝对路径依赖



保留原有评分逻辑，仅做最小侵入改造。



\### 3.3 目录规范（Project Structure）

严格按照项目要求搭建：

\- 测试用例：`tests/e2e/models/configs/`

\- 脚本：`tests/e2e/models/`

\- 教程文档：`docs/source/tutorials/models/`



\### 3.4 教程文档编写（Tutorial）

撰写完整的 `phi3-5-moe-test-guide.md`，包含：

\- 环境准备

\- 文件结构说明

\- 执行步骤

\- 用例扩展方式

\- 结果分析



\## 4. 可复用模式（reusable Patterns）

\### 4.1 大模型测试配置化模式

Skill 模式：`YAML 配置文件 + Python 执行脚本 + 独立加载模块`

优势：修改用例无需改动业务代码，提升可维护性。



\### 4.2 路径自适应规范

使用 `os.path.dirname(os.path.abspath(\\\_\\\_file\\\_\\\_))` 获取脚本目录，动态拼接配置路径。

适用于所有 AI 模型项目的脚本路径管理。



\## 5. 验证与检查（validation）

执行以下命令校验技能结构：



```bash

skills-ref validate ./phi3.5-moe-instruct-testing

确保：

- 目录名称与 skill name 一致

- 无大写字符

- 无连续连字符

- frontmatter 完全合法


