# CANN 9.2.0.beta2 A3 实验

本目录仅用于实验 PR，验证两次 Nightly 在 `aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize` 中崩溃的问题。

工作流使用独立 A3 job，在临时容器的随机 `/opt/cann-920beta2-experiment.*` 目录内同时安装 toolkit 与 A3 ops 9.2.0-beta.2。安装包来自此前升级 ops 的 PR #15643 使用的 CANN 9.2.T3 目录，输出下载包 SHA256、安装元数据和实际加载的库路径。

模型代码固定为第二次失败 job 的 vllm-ascend 提交 `60162f1046750d35c325d0449fcbbcd4af28d488`。继承镜像中的 Python 依赖，但强制检查 torch_npu 为 2.10.0.post4、vLLM 为 0.27.1；若镜像漂移，实验会明确失败。公共基础镜像仍为可变标签，不能把本实验当作历史镜像的完全复刻。

先运行 BF16 `[1/48/4096,7168]` 单路量化 eager 调用，再使用原 DeepSeek-V3.2-W8A8 TP8/DP2 配置，分别开启和关闭 `fuse_norm_quant` 检查服务是否启动到 `/health`。两组均使用各自全新的缓存目录，测试后停止本组服务。该门禁只验证启动阶段，不声明模型精度或性能通过。

该 PR 新增的 `pull_request` 工作流需经过仓库正常的 fork 工作流审批。现有 `/nightly` 命令使用主分支工作流并传入 `skip_build_image=true`，不会自动构建本 PR 的实验环境。

查看 `cann-920beta2-a3-*` artifact 中的 `experiment.log`、`cann-install-info.txt`、`active-cann-paths.txt`、`eager-loaded-libraries.txt`、`fusion-on-server.log`、`fusion-off-server.log` 和 `result.json`。cold-cache 成功只说明该受控组合未复现；认定缓存兼容问题仍需在相同环境对旧缓存做 A/B。
