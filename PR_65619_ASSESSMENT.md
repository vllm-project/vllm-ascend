# vLLM PR #65619 Ascend 适配评估

评估日期：2026-11-09。Ascend 基线：28e9e97ba。

## 结论

需要独立实现内存优化，但不是上游接口变更导致的必需兼容修复。已实现 target 层的构造期共享，DBO 开启时关闭共享，MTP 保留独立存储。

上游 PR https://github.com/vllm-project/vllm/pull/65619 只改变 NVIDIA Qwen4Exp 路径。Ascend model.py 继承 AMD Qwen4ExpModel，qsa.py 继承 AMD Qwen4ExpQSAAttention，因此不自动继承优化。本地 AMD qsa.py 仍在每个 attention 中 register_buffer，形状为 max_num_batched_tokens × indexer.output_width，dtype int32。

## 实现约束

- 应在模型构造时提供共享 buffer，避免先分配所有层再重绑造成初始化峰值；保留 persistent=False 和正确设备。
- 每个 PP rank 独立分配，只覆盖本 rank 的 QSA 层；无 QSA 层不分配。
- 需要验证 DBO/微批并发不能覆盖其他层仍在使用的索引。不能直接套用 NVIDIA 路径拒绝 DBO 的假设。
- AMD MTP 的 compact_topk_indices 遍历各 attention，且 set_skip_topk 会复用 step-0 索引；多个 draft 层共享可能覆盖各自历史索引，重复 compact 同一 storage 也可能破坏行映射。应保留独立 draft 存储，直到验证这些语义。
- 不改变选择算法、排序、有效 token 范围及 attention 数学计算。

## 验证要求

单测应覆盖 storage 别名、PP 局部层、无 QSA、设备/dtype/shape、非持久 buffer、索引覆盖范围、MTP 多步及非平凡行压缩映射。NPU 验证需覆盖 eager/图回放、prefill/decode、prefix cache、MTP 开关和多步，比较索引与输出，以及端到端精度、接受率、TTFT/TPOT 和峰值/常驻内存。

理论 target buffer 节省为 (本 rank QSA 层数 - 1) × max_num_batched_tokens × output_width × 4 字节。上游 GPU 测量不能作为 Ascend 实测。

## 当前限制

在线检索未找到对应 Ascend PR，但 GitHub PR 列表读取失败，shell GitHub API 请求因网络失败，不能证明不存在相关 PR。本机 Python 无 torch，尚不能运行模型单测；未进行 NPU 精度和性能测量。运行时代码已修改，但尚未经过 NPU 验证。构造时上游仍短暂分配当前层 buffer，随后立即重绑；不是完全消除临时分配的实现。

## PR #21295 rebase 验证（2026-11-23）

- 基于用户提供的本地 main `985d07c84`，通过 `git rebase` 重放两个 feature 提交；原分支备份为 `backup/1109_1-before-rebase-20261123`。
- 冲突位于 `qsa.py` 的 lightning indexer 导入。保留 main 的 `QSAIndexerMetadataProtocol` 和 `qsa_select_paged_tokens_lightning_v2`，追加共享 buffer helper；不回退 main 的选择器、缓存布局或 attention 路径。
- target 在构造作用域内重绑兼容 buffer；DBO 禁用共享，独立构造的 MTP 不进入 target 作用域。共享逻辑不增加推理热路径操作，也不改变索引选择和 attention 数学计算。
- CPU 验证：Python 3.12 / PyTorch 2.2.2，`test_topk_buffer.py` 和 `test_lightning_indexer_v2.py` 共 17 项通过。额外覆盖禁用的嵌套作用域恢复及 CPU/meta 设备隔离。
- 执行方式：`PYTHONPATH=/private/tmp/dspine-test-deps /Users/xk/.local/bin/python3.12 /private/tmp/qsa-rebase-21295/run_tests.py`。临时 runner 使用 `pytest --noconftest`，仅将父包设为带真实源码路径的 namespace，加载真实 helper/selector；选择器测试按原测试替换 NPU 算子。该结果不代表完整插件启动或 NPU 集成测试通过。
- Ruff 检查、格式检查及 `git diff --check` 通过。`bash format.sh ci` 因未安装 pre-commit 未执行；系统 pytest 入口退出码为 139，随后改用上述独立 CPU 环境。
- 本次未执行 Ascend 硬件测试。仍需在同一硬件和负载下比较共享前后的 eager/图回放、prefill/decode、prefix cache、DBO、MTP 多步、输出精度/接受率、TTFT/TPOT、吞吐和峰值/常驻内存。上面的静态分析与 CPU 测试不能作为这些指标已达标的证据。
