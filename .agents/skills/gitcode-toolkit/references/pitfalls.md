# Git 操作易错点

常见 Git 操作错误及正确做法对照。

---

## 易错点对照表

| 问题 | 正确做法 | 错误做法 |
|------|----------|----------|
| refs 路径 | `refs/merge-requests/{n}/head`（GitCode） | `refs/pull/{n}/head`（GitHub） |
| 比较基准 | `git diff $MERGE_BASE pr_xxx` | `git diff base_branch pr_xxx` |
| merge-base 失败 | `git fetch --deepen=500` 后重试 | 直接使用 base_branch |
| 克隆深度 | `--depth=500`（标准）/ `--depth=200`（轻量） | 不指定 depth（全量克隆太慢） |
| PR 文件列表 | 优先用 `git diff --name-status` | 仅依赖 API（可能认证失败） |
| triple-dot vs double-dot | `git diff A...B`（A 和 B 的共同祖先到 B） | `git diff A..B`（A 到 B 的差异） |
