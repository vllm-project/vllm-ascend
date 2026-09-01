# 上游来源

本目录为外部仓库样例的本地拷贝，仅用作 `ascendc-direct-invoke-template` skill 在 Kirin 平台开发场景下的参考工程模板。

| 项 | 值 |
|----|----|
| 上游仓库 | https://gitcode.com/cann/cann-recipes-harmony-infer |
| 上游路径 | `ops/ascendc/AddKernelInvocation/` |
| 拷贝时 commit | `2006a857b301893453873919f6f5c01fdccf70c1` |
| 拷贝日期 | 2026-05-23 |
| 上游 License | Apache License 2.0 |

## 与上游的差异

- **已剔除运行时产物**：`build/`、`add_sim`（编译后的可执行文件）。
- **Kirin 平台适配修改**：
  - `add_custom.cpp:20`：`USE_CORE_NUM` 从 8 改为 1（Kirin 单核）
  - `add_custom.cpp:77-78`：`TQue` 队列深度参数从 `BUFFER_NUM` 改为 `1`
  - `main.cpp:29`：`blockDim` 从 8 改为 1（与 USE_CORE_NUM 一致）
- **新增文件**：`README.md`，替换上游原始 README，改为 Kirin 适配说明文档（含目录结构、运行方式、基于模板开发新算子的完整指引）。
- **其余源文件未做修改**：`run.sh`、`data_utils.h`、`CMakeLists.txt`、`cmake/*`、`scripts/*`、`input/.keep`、`output/.keep`。

## 同步策略

本地拷贝不会自动跟随上游更新。若需要同步：

1. 从上游对应 commit 重新拷贝（参考本文件「拷贝时 commit」字段升级到新版本）。
2. 重新应用「与上游的差异」中列出的 Kirin 适配修改（USE_CORE_NUM、TQue 队列深度、blockDim）。
3. 更新本 README.md 的 Kirin 适配说明（如有新内容需补充）。
4. 在 CHANGELOG 记录同步动作与新 commit 哈希。
5. 同步时再次剔除 build/、可执行文件、`*.bin`。

## License

上游为 Apache License 2.0。本目录文件均沿用该许可。详见 [上游 LICENSE](https://gitcode.com/cann/cann-recipes-harmony-infer/blob/master/LICENSE)。
