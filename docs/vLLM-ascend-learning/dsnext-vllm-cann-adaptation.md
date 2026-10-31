# DSNext 三个 PR 的 vllm_cann 适配

先看修改前后对比图：[三个 PR 修改图解](three-pr-visual-guide.md)。

基线：主仓 `rfc/vllm_cann`，`0040652a66d31a2b277588ac90d7d37e92757bc5`。

当前合入分支：`integrate/dsnext-megamoe-vllm-cann`。

三个来源提交已经保留原作者、作者时间和提交说明，并记录来源 SHA：

| 来源 | 原始提交 | 当前提交 |
| --- | --- | --- |
| !18：DSV4.1 A5 MegaMoE | `df1286cca` | `1f2c71ce1` |
| !31：混合 PD 容量修复 | `bca6b8ff8` | `122552f76` |
| !51：共享专家融合 | `8d0686064` | `0cd155130` |

学习原始PR18的方案、17文件差异和当前实现的对应关系，见 [PR18学习详解](pr18-megamoe-design-and-code-walkthrough.md)。原始PR18不包含PR31的无KV配置容量修复，也不包含PR51的共享专家融合。

## 适配内容

- 原始 PR 引入的独立后端与 RFC 已有实现重复，且直接迁移后引用了不存在的后端属性。适配为扩展 RFC 的 `FusedMC2CommImpl`，沿用现有 prepare/finalize、通信组及量化方法提供权重的接口。
- 补齐 `shared_weights` 从 MoE runner、量化方法到融合算子的传递，将融合状态接入 `RoutedMoEMilestones`。融合后的总输出放入共享输出槽，路由槽返回零，避免重复相加及路由专属缩放影响共享贡献。
- 检查算子是否明确暴露六个共享权重参数，验证共享权重形状、连续性、scale 类型及独立 FP8 权重类型。不兼容配置保留独立共享 MLP；权重重载后清空连续权重缓存，不兼容布局提前回退。
- A5 输入容量受配置上限除以 EP 大小和当前 PD 角色的执行容量共同约束。通信选择、缓冲区分配和执行检查使用一致的容量；超限时回退。保留 RFC 的接收缓冲区和 HCCL teardown/restore 逻辑。
- 补齐 `is_dummy_run` 参数声明及调用，使空闲 DP rank 不放大实际 batch，也不单独关闭实际任务的图执行；全 dummy 的图捕获仍统一协调。
- 保留 RFC 的 `DeviceOperator.compute_gate_logits`，融合共享专家也经由该接口计算内部路由。保留 RFC 的硬件能力选择和 vision/hash 算子，只有 hash 查表路径转换 ID 并清理 padding。
- 更新既有 forward context、MoE 通信和 model runner 测试及配置说明；新增 29 个无 NPU 回归测试，其中三项检查 RFC 内部路由、预处理 FP32 输入复用和自定义 gate 回退。

## 验证及限制

已通过：29 个主机回归测试、18 个 Python 文件编译、Ruff 0.14.0 lint/format、布尔 context manager AST 检查、Git 空白检查及冲突标记检查。

额外与 RFC 基线核对：13 个定义的 AST 保持一致，包括配置类、CANN 内部路由、共享多流输入准备、量化权重加载及 HCCL 生命周期；device adaptor、prepare/finalize、parallel state 和 distributed patch 四个文件保持一致。模型、attention、KV transfer、core 和 C++ 算子目录无改动。

主机测试使用生产函数和显式依赖替身，不执行 NPU 算子。当前环境无 `torch`、`torch_npu`、`vllm`、`pytest` 和 Ascend NPU，因此完整运行时测试、实际推理精度和性能仍需验证。

`bash format.sh ci` 因 Windows 沙箱共享内存映射权限错误无法运行；未报告完整格式流程或 Markdownlint 通过。

## 创建适配提交

文件修改已完成，但本会话不能写入原仓库 `.git`。在用户自己的 PowerShell 运行：

```powershell
& 'D:\code\vllm-ascend\commit-vllm-cann-adaptation.ps1'
```

脚本只暂存指定的 20 个适配文件，创建带 sign-off 的第四个提交。旧清理脚本、v30 工作副本、本文和提交脚本不进入提交。当前尚未推送 RFC 合入版本。
