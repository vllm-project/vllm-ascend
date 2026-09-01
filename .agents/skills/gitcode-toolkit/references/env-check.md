# 环境预检（Step 0）

GitCode 相关 skill 在执行任何业务步骤之前，必须先完成环境预检。预检失败时应**立即**停止并通过 AskUserQuestion 询问用户，而不是带着不完整的环境继续往下走（否则常见症状：API 401、克隆卡死、行号偏移等问题会延迟暴露，浪费排查成本）。

预检遵循一个原则：**能本地解决的不打扰用户，缺了用户才知道的关键输入立刻问**。

> **脚本支持**：可通过 `bash scripts/preflight.sh [--skip-git-author]` 一键执行预检，输出结构化 JSON 报告。详见 [SKILL.md 脚本工具](../SKILL.md#脚本工具)。

---

## 预检项

### 1. Token（必检）

按以下顺序探测：

1. 用户当前消息中是否直接提供了 token（关键词：`token=...`、`access_token=...`、形如 `gitcode_pat_*` 或 32+ 位字母数字串）
2. 环境变量 `GITCODE_TOKEN`：`echo "${GITCODE_TOKEN:+set}"`（只判断是否存在，不打印明文）
3. 都没有 → 通过 AskUserQuestion 询问用户

询问示例：

```
问题: 未检测到 GitCode Token，需要你提供以继续。
选项:
  - 我现在提供 token（在下一条消息中输入）
  - 已配置环境变量，让我重试一次
  - 取消本次操作
```

拿到 token 后**只在当前会话临时使用**，不写入任何文件、不写入日志明文。日志中只记录"已获取 token（来源：env / user / arg）"。

### 2. Git 与基础工具（必检）

```bash
command -v git >/dev/null 2>&1 || echo "MISSING: git"
command -v curl >/dev/null 2>&1 || echo "MISSING: curl"
command -v python3 >/dev/null 2>&1 || echo "MISSING: python3"
```

任一缺失 → AskUserQuestion 告知用户需要安装，并询问是否继续。

### 3. 临时目录（必检）

确认 `/tmp` 可写：

```bash
test -w /tmp || echo "MISSING: /tmp not writable"
```

不可写 → AskUserQuestion 询问替代路径（如 `~/tmp`）。

### 4. 输出目录（仅 pr-to-design-doc / 需要落盘的 skill）

确认目标输出目录（如 `./docs/`、用户指定路径）的父目录存在且可写。不存在则 `mkdir -p` 创建，无权限则 AskUserQuestion 询问替代路径。

### 5. Git 提交用户信息（创建 commit 或提交 PR 的 skill）

适用范围：① 会调用 `git commit` 的 skill（如 `gitcode-issue-handler`）；② 走「PR 创建工作流」提交 PR 的流程（见 SKILL.md「PR 创建工作流」Step 5）。这两类都依赖正确的 git 提交身份（`user.name` / `user.email`）。

会调用 `git commit` 的 skill 必须在 Step 0 阶段确认 `user.name` 和 `user.email` 已配置，否则 `git commit` 会在后续步骤才报错（`Author identity unknown`），白白浪费一轮 clone+改代码+跑测试的工作量——越早暴露越好。提交 PR 的流程虽然 `git push` 不会因身份缺失而失败，但 commit author 是公开字段，配错人后难以补救，因此同样需要在推送前校验。

读取顺序：**local（项目级）→ global → 询问用户**。服务器多人共用时，global 配置往往不是当前项目期望的身份；更常见的做法是在项目级别设置 `user.name` / `user.email`。

```bash
NAME=$(git config --local user.name 2>/dev/null)
EMAIL=$(git config --local user.email 2>/dev/null)
if [ -n "$NAME" ] && [ -n "$EMAIL" ]; then
  GIT_AUTHOR_SOURCE="local"
else
  NAME=$(git config --global user.name 2>/dev/null)
  EMAIL=$(git config --global user.email 2>/dev/null)
  if [ -n "$NAME" ] && [ -n "$EMAIL" ]; then
    GIT_AUTHOR_SOURCE="global"
  fi
fi
```

- local 两项都已配置 → 标 `GIT_AUTHOR_SOURCE=local`；后续在工作目录上直接使用（若工作目录就是当前目录则已生效；若 clone 到新目录则需写入 local）
- local 缺失但 global 两项都已配置 → 标 `GIT_AUTHOR_SOURCE=global`；后续克隆出的工作目录会**自动继承** global 配置，**无需**再 `git config` 写到 local
- 两者都缺失 → AskUserQuestion 询问用户

询问示例（仅 local 和 global 都缺失时触发）：

```
问题: 未检测到 git 提交用户信息（local / global user.name / user.email 均未配置），请提供：
选项:
  - 用我下面提供的 name 和 email（在下一条消息中给出）
  - 用 fork_owner 拼出一个占位（不推荐，作者归属会失真）
  - 取消本次操作
```

拿到用户提供的值后**只在工作目录上写**，**禁止改 `~/.gitconfig` 全局**：

```bash
# 仅当 Step 0 走"用户提供"分支时执行；走 local / global 分支跳过
git -C "$WORK_DIR" config user.name  "$NAME"
git -C "$WORK_DIR" config user.email "$EMAIL"
GIT_AUTHOR_SOURCE="user"
```

当 `GIT_AUTHOR_SOURCE=local` 且工作目录是新 clone 的目录（非当前目录）时，需要将 local 值写入新目录：

```bash
# 仅当 GIT_AUTHOR_SOURCE=local 且 WORK_DIR ≠ 当前目录时执行
git -C "$WORK_DIR" config user.name  "$NAME"
git -C "$WORK_DIR" config user.email "$EMAIL"
```

理由：用户的全局 git 配置可能服务于多个项目和身份，skill 不应擅自覆盖；用户主动提供的值是一次性使用，只配 local 不污染全局。

预检报告里展示读到的 `Name <email>` 及来源（local / global / user），方便用户一眼确认是否是本次想用的身份（commit author 本就是公开字段，预检阶段透明展示反而避免事后才发现写错了人）。

---

## 预检报告格式

预检完成后，向用户输出一段简短的可视化报告（不要 AskUserQuestion，除非有项目失败）：

```
环境预检
  ✅ Token: 已获取（来源：env GITCODE_TOKEN）
  ✅ Git: git version 2.39.2
  ✅ Curl: 可用
  ✅ Python: python3 3.10.12
  ✅ /tmp: 可写
  ✅ Git author: pingchuantang <pingchuantang@gitcode.com>（来源：local / global / user，仅 gitcode-issue-handler 等创建 commit 的 skill 才检）
预检通过，开始执行 ...
```

任一项失败时，必须**先通过 AskUserQuestion 解决该项**，不要带着失败项继续。

---

## 何时复用预检

- 同一会话内首次执行 GitCode skill 时必检
- 后续 skill 调用如果是同一 token、同一会话，可跳过 token 询问，但仍打印一行"沿用上次会话 token"以便用户知情
- 如果中途出现 401 / 403，立即重新走预检的"询问用户提供新 token"分支，而不是反复重试

---

## 反模式

- ❌ 跳过预检直接 `curl ... ?access_token=$GITCODE_TOKEN`，让 401 自然暴露
- ❌ 把多个预检失败项打包到一条 AskUserQuestion 里（违反"一次只问一个"的原则）
- ❌ 在日志中打印 token 明文
- ❌ 自动写入 token 到 `~/.bashrc` 或任何持久化文件
- ❌ 创建 commit 类 skill 跳过 git author 预检，等到 `git commit` 才报 `Author identity unknown`——clone+改代码+跑测试的工作量已经废了
- ❌ 修改全局 `~/.gitconfig` 的 user.name / user.email——会污染用户的其他项目身份；只在工作目录用 `git -C <work_dir> config` 设置
- ❌ **从 fork_url / 用户名等地方"脑补"一个身份**塞进 commit（典型反例：`git -c user.name='pingchuantang' -c user.email='pingchuantang@gitcode.com' commit ...`）。即便看起来能跑通，最终 PR 上的 commit author 是错的——可能挂错人头、与用户实际身份不符。身份只能来自三个来源：local 配置、global 配置 或 用户在 AskUserQuestion 里提供，没有第四条路。
- ❌ **只读 global 跳过 local**——服务器多人共用时 global 往往不是当前项目身份，必须先查 local 再 fallback 到 global。
- ❌ 用 `git commit -c user.name=...` / `--author="..."` 这类 inline 覆盖来绕过 Step 0 预检——任何时候 `git commit` 命令都不应带身份相关的 flag，让它读 work_dir / global 配置即可。
