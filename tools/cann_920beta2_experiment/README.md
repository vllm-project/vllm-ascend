# CANN 9.2 beta2 A3 修复验证实验

本目录用于实验 PR #15849，复现并验证 `aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize` 的 tiling SIGSEGV 修复。

工作流复用 `linux-aarch64-nightly-a3-16` runner，在临时容器内安装匹配的 toolkit/A3 ops 9.2.0-beta.2。公共镜像和正式 Nightly 配置不变。CANN 包来自已有 ops 升级实验使用的 9.2.T3 目录，并记录下载包哈希、安装元数据及生效路径。

模型代码固定为 `60162f1046750d35c325d0449fcbbcd4af28d488`，检查 torch_npu=2.10.0.post4、vLLM=0.27.1+empty。基础镜像标签可变，版本漂移会显式失败。

## 本轮步骤

准备阶段在安装 CANN 之前检查构建工具，安装和检查共用 `git`、`cmake`、`g++`、`make`、`pigz`、`dos2unix`、`unzip`、`curl`、`patch`、`pkg-config` 清单，在本 job 的临时容器内一次安装全部缺项，记录工具路径及版本，并拉取、应用和核验固定 CANN 源码补丁。后续编译再次核验源码树，提前暴露缺工具或源码下载失败。

1. 在原包下执行 BF16 `[1,4]`、带 beta、无 smooth 的调用，要求复现 SIGSEGV。
2. 拉取 ops-nn 9.2 beta2 固定基线 `30ef7dd56`，应用可选输入描述信息修复及回归用例。校验完整源码树为 `c41d331cdab95233834dbca87e0ada238251a5f0`，与已在 A2 验证的修复提交 `9f85253fa` 一致。
3. 按 A3 型号 `ascend910_93` 编译自定义包，在本 job 独立目录安装。记录源码树、依赖提交、安装包与动态库哈希。
4. 验证修复包的带 beta eager/npugraph_ex 调用，rows=4096/48/1、hidden=7168；要求成功且有效输出有限，并核验新库实际加载。
5. 提前上传修复包和单算子证据，再使用原 DeepSeek-V3.2-W8A8 TP8/DP2 配置，保持 Ascend norm-quant 融合开启，从全新缓存启动模型。
6. `/health` 成功后发送一个 8-token completion 请求。核验至少 16 个 worker 的融合配置、带 beta 的成功调用与新库哈希。

原包复现以 faulthandler 段错误栈和同 PID 的 beta-only 实际调用记录共同判定。若段错误后进程未退出，控制器清理该探针进程组，并记录真实终止码；普通超时或无对应调用证据的异常仍失败。修复包必须正常退出且通过后续检查。

本轮验证模型启动及一次真实推理，不运行 AIME/GSM8K 完整精度、性能基准。A2 编译的 `ascend910b` 包不直接用于本轮 A3 验证。

## 诊断与历史更正

Ascend 开关必须设置在 `additional_config.ascend_compilation_config.fuse_norm_quant`。旧 bb8587b 实验误设 upstream pass_config，不能作为关闭 Ascend 融合的对照；修正后的 81bd622 已验证关闭融合可启动。

诊断只记录每个进程首次真实 V2 调用及首次成功结果，避免对已修复的大模型每次调用反复写堆栈。保留原生错误日志、实际融合配置、加载库及返回后的哈希核验。检测到原生段错误时及时结束本实验进程组。

该 fork PR 的工作流仍遵守上游正常审批和排队规则。现有 `/nightly` 使用主分支工作流，不会自动应用这里的修复包。

查看 `cann-920beta2-minimal-*` 和 `cann-920beta2-a3-*` artifact：包版本、fix-build.log、fix-build-metadata.json、fix-packages、stock/fixed-beta-result.json、fusion-on-server.log、fusion-on-diagnostics、completion-smoke.json、result.json。
