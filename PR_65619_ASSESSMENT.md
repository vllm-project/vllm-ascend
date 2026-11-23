# PR #21295：QSA Top-K buffer 共享评估

评估日期：2026-11-24。Rebase 基线为本地 Ascend main `b776261e3`。构造保护以本地 vLLM `c410543d9f` 的微批配置属性进行隔离验证。

## 必要性与实现边界

该版本 NVIDIA 路径已有共享优化，但 Ascend 继承的 AMD 路径仍在每个 QSA attention 中分配 `max_num_batched_tokens × indexer.output_width` 的 int32 buffer。因此本 PR 是独立的内存优化，不是接口兼容所必需的修复。

- 在单次 target 模型构造作用域内共享同 shape、stride、dtype、device 的 buffer；异常退出会恢复上下文，不跨模型共享。
- PP 只构造本 rank 的真实层；无 QSA 层时不额外分配。每组同规格 buffer 的理论常驻内存节省为 `(层数 - 1) × max_num_batched_tokens × output_width × 4` 字节。
- PCP 保持私有 buffer。使用上游 `parallel_config.use_ubatching` 关闭微批场景的共享，同时覆盖 DBO 和 `ubatch_size > 1`。原先只检查 `enable_dbo` 的实现遗漏了后一种模式。
- MTP 独立构造，不进入 target 的共享作用域。AMD MTP 存在 `set_skip_topk` 跨步索引复用和逐层 `compact_topk_indices`，不能直接照搬 NVIDIA target/draft 共享策略。
- 保留上游非持久 buffer 注册以及 main 的 LightningIndexerV2 路径；不改变选择算法、有效 token 范围及 attention 数学计算。
- ContextVar 只在构造时使用，存储池在退出作用域时释放引用；层继续持有共享 tensor。推理热路径不查询共享池，不增加设备同步、复制或计算。
- 上游构造函数仍短暂分配当前层 buffer，再立即重绑。本实现降低常驻内存，但不声称完全消除临时分配。为避免全局拦截 tensor 分配或 buffer 注册，暂不扩大适配范围。上游 AMD 提供共享 buffer 参数后，应迁移至显式参数并移除此 helper。

## 本地验证

环境：Python 3.12、CPU PyTorch 2.2.2。本机缺少完整 vLLM/NPU 运行依赖。

- helper 与 LightningIndexerV2 CPU 测试：20 项通过。覆盖共享与隔离、异常和嵌套作用域、shape/stride/dtype/device、非持久 buffer，以及 0/1/3/8 个有效 token 的顺序消费结果与独立 buffer 逐位一致。
- 构造路径回归测试：7 项通过。覆盖 DBO 开关、ubatch_size 0/1/2/4、PCP 的私有存储和缓存配置、多个 target 模型隔离和独立 draft attention。测试 mock 上游构造开销，执行 Ascend target/QSA 构造函数。
- 本地隔离 runner 从真实源码提取上述构造函数和指定 vLLM 的 `use_ubatching` property，在 CPU 基类上执行同一回归测试；这不等同于完整模块导入或完整模型构造。恢复旧 guard 的对照实验恰好在 DBO 关闭且 ubatch_size 为 2/4 时失败（2 failed、4 passed），截至首次修复，新 guard 为 6 passed；本次新增 PCP 验证。
- Ruff lint、格式检查及 `git diff --check` 通过。`bash format.sh ci` 因缺少 pre-commit 未能执行。

本地复现命令（临时依赖与 runner 仅存在于当前机器）：

```bash
PYTHONPATH=/private/tmp/dspine-test-deps /Users/xk/.local/bin/python3.12 /private/tmp/qsa-main-20261124/run_tests.py
PYTHONPATH=/private/tmp/dspine-test-deps /Users/xk/.local/bin/python3.12 /private/tmp/qsa-main-20261124/check_constructors.py
```

完整 CPU CI 应在匹配的依赖环境中运行：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -q tests/ut/models/qwen4_exp/test_topk_buffer.py tests/ut/models/qwen4_exp/test_lightning_indexer_v2.py tests/ut/models/qwen4_exp/test_qsa_upstream_abi.py
bash format.sh ci
```

## 待完成的硬件验收

CPU mock 与静态审查不能证明 NPU 精度、图回放正确性或性能达标。合入前必须在相同硬件、软件、模型和负载下对比共享前后：

- eager 与图回放，prefill/decode、chunked prefill、prefix cache；确认下一层写入前上一层所有消费者已完成读取。
- MTP 开关和多步，比较输出、接受率；DBO 与显式 ubatching 确认保持私有存储。
- 多卡 TP/PP 与各 rank 的 buffer 存储关系。
- 端到端精度、TTFT/TPOT、吞吐、峰值与常驻内存；记录硬件、版本、配置、命令及结果。

本次没有 Ascend 实测，不能将理论内存节省或 CPU 测试报告为 NPU 性能和精度结论。
