# Agent Workspace

Agent 工作区，存放 AI agent 辅助开发的输入素材、输出产物，以及跨机器的 git 同步脚本。

```
.workspace/
├── scripts/                   # 工具脚本
│   ├── setup-workspace-branch.sh  # 一次性初始化（创建 orphan 分支 + worktree）
│   └── sync-outputs.sh           # 日常 push/pull/status
│
├── inputs/                    # 输入侧：给 agent 的参考素材
│   ├── prompts/               # 可复用的 prompt 模板
│   └── experience/            # 经验文档（踩坑记录、适配 pattern）
│       └── git-sync-workspace.md  # Git 同步使用指南
│
└── outputs/                   # 输出侧：agent 生成的分析和报告
    ├── model-analysis/        # 模型 NPU 适配分析
    └── code-review/           # 代码审查记录
```

## 初始化

每台机器执行一次：

```bash
bash .workspace/scripts/setup-workspace-branch.sh
```

## 同步产出

服务器和本地之间通过独立的 `workspace-outputs` orphan 分支同步 `outputs/`：

```bash
bash .workspace/scripts/sync-outputs.sh push     # 本地 → 远端
bash .workspace/scripts/sync-outputs.sh pull     # 远端 → 本地
bash .workspace/scripts/sync-outputs.sh status   # 查看状态
```

详细说明见 `inputs/experience/git-sync-workspace.md`。

## 使用方式

- **inputs/** — 把反复使用的 prompt、踩坑经验写成 markdown，下次让 agent 参考这些文件
- **outputs/** — agent 产出的分析报告放这里，方便回顾和分享

## 与 .agents/skills 的区别

- `.agents/skills/` — 给 agent 用的**可执行流程**（SKILL.md 定义了操作步骤）
- `.workspace/inputs/` — 给 agent 用的**参考知识**（经验、模板，非结构化）
- `.workspace/outputs/` — agent 的**工作产物**（分析报告、审查记录）
