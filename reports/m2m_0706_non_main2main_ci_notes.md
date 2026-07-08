# m2m-0706 非 main2main CI 问题说明

## 背景

当前 PR 的代码目标是 main2main 适配：

- vLLM main commit: `ba22152096b2484faa3579624a253d54804d876d`
- vLLM release tag: `v0.23.0`

本 PR 中保留了一项 CI 行为增强：当某个 E2E target 失败时，继续执行同一 job 中剩余 target，最后统一失败。这个改动不是 vLLM 上游接口变化导致的 runtime 适配，但对 main2main PR 有排障价值，可以一次性暴露更多 E2E 问题。

## 保留的 CI 行为增强

| 文件 | 改动 | 保留原因 |
|---|---|---|
| `.github/workflows/_selected_tests.yaml` | `continue_on_error` 默认改为 `true`，并把该参数传给脚本 | 让 main2main PR 的 E2E 在首个失败后继续跑，便于收集完整失败面 |
| `.github/workflows/pr_test.yaml` | matrix 增加 `fail-fast: false` | 避免一个 vLLM 目标失败后取消另一个目标或其它硬件分片 |
| `.github/workflows/scripts/run_selected_tests.sh` | 增加 `--continue-on-error`，记录失败日志，最后统一 `exit 1` | 保留最终失败信号，同时输出更多 target 的真实结果 |

## 已处理的非 main2main CI 问题

| 问题 | 现象 | 原因 | 当前处理 |
|---|---|---|---|
| markdownlint Node 环境安装失败 | `pre-commit` 安装 `node_env-22.17.1` 时找不到 `npm/man/man1` | `.pre-commit-config.yaml` 固定 `language_version: 22.17.1` 触发 CI 容器 nodeenv 安装问题 | 已删除 `language_version: 22.17.1`，让 pre-commit 使用默认 Node 环境 |

## 不属于本 PR 范围的 CI/测试适配

| 项目 | 为什么不属于当前 PR | 后续建议 |
|---|---|---|
| `v0.24.0` GPT-OSS E2E skip | 当前 PR 不升级 release tag 到 `v0.24.0` | 后续单独升级 `v0.24.0` 时，再决定 skip 或 backport vLLM GPT-OSS loader 修复 |
| `v0.24.0` release tag 矩阵适配 | 当前 PR release tag 保持 `v0.23.0` | 后续独立 PR 中处理 `v0.24.0` 的 InputBatch、rejection sampler、DeepSeekV2 hidden_size、DFlash kernel 等差异 |

## 后续建议

1. 当前 main2main PR 可继续保留“失败后继续执行”的 CI 能力，直到本轮 main2main E2E 问题收敛。
2. 若维护者希望 PR 范围更纯净，可在问题收敛后把 CI 行为增强拆成单独 PR。
3. `v0.24.0` release tag 升级应单独开 PR，不要和本次 `ba221520...` main commit 适配混在一起。
