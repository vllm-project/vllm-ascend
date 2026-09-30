# GLM-5.3-Flash KPool 后处理 A/B 实验

## 实验范围

- 基线提交：`d230e9c5fb6ba79365e3e94bfdbae2e6b9c5bdd5`。
- 实验分支：`perf/glm5next-kpool-fused-postprocess-clean`。
- 只优化 KPool top-k 之后的索引处理；不改打分、top-k 选择、QKV 投影或 SFA 输出投影。
- 这是待 NPU 验收的实验实现，不包含已测得的加速结论。

原路径通过多个张量操作展开 pool 编号、拼接尾部，然后外层再次生成尾部并用 scatter 修正短序列的有效前缀，最后屏蔽 padding 行。实验路径在每个 token chunk 的 top-k 后启动一次 `_glm5_next_kpool_postprocess_kernel`，一次写出最终 int32 索引，保留原有排序、无效值、尾部覆盖和 padding 语义。无 pool 时也使用融合后处理。共享 top-k、无需重新选取索引的层仍只更新缓存。

模型包装层启用 `compact_indices=True`。底层函数默认保留 `False`，供现有直接调用者和 A/B 脚本使用；原 `append_causal_tail` 也保留为参考。融合路径不执行原后处理链。未增加环境变量。

本分支从官方 `vllm-project/vllm-ascend` 的上述提交新建，仅迁移本次后处理优化，不继承旧 fork 的额外提交。保留官方 TopKV2 安全修复：top-k 前将 NaN/-inf 转为 FP32 最小有限值；融合后处理使用同一个 sentinel 过滤无效项。该修复不是冗余操作，本实验不移除。

## 先验证正确性

使用安装了匹配 vLLM、torch-npu、Triton Ascend 和此分支的 NPU 环境，在仓库根目录执行：

```bash
pytest -q tests/ut/models/test_glm5next_kpool_model_backend.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_glm5next_kpool_postprocess.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_glm5next_pool_key_indexer_triton.py
```

新增测试覆盖 pool size 1/4/16、top-k 32/2048、零 pool、无效分数、短请求、分组边界、长请求、padding、非零 chunk 偏移，以及图回放时更新 positions 和有效行数。另将完整 indexer 的融合输出与原 indexer 加外层修正逐元素比较。旧测试继续覆盖原始固定尾列接口。

任一正确性或编译测试失败，应停止性能结论采集，保留报错用于修正实现。

## 同一进程对照与 profiling

```bash
python -m benchmarks.glm5next_kpool_postprocess --rows 32 --pools 1024 --iterations 100 --trace-dir ./kpool-ab-eager
python -m benchmarks.glm5next_kpool_postprocess --rows 32 --pools 1024 --iterations 100 --graph --trace-dir ./kpool-ab-graph
```

脚本先验证两条完整 indexer 路径输出一致，再分别预热、采集五组平均每次调用耗时，并分别导出 `baseline` 和 `fused` 的 trace。计时为同步边界内的 wall time，包含主机调度及设备执行，不能直接解释为单 kernel 耗时。profiler 在计时结束后开启，避免其开销混入计时。

trace 目录必须是新的目录，避免覆盖先前结果。参数 `--rows` 为 query token 数，`--pools` 为缓存中的 pool 数。建议覆盖 rows 1/32/256、pools 0/4/512/8192，以及 eager/graph 两种模式。脚本是合成输入微基准，不替代真实模型 prefill/decode 和端到端验证。

## 完整模型前后对照

基线使用上述固定提交，实验版使用该分支最终提交。分别启动独立服务进程，确认安装路径和实际 commit，保持模型权重、请求集、并发、输入输出长度、TP、量化配置、编译和图模式完全一致。预热并重复采集，避免两个服务同时争用同一 NPU。

比较 KPool 后处理总耗时、算子数量、整个 indexer 耗时以及端到端延迟/吞吐；不要只比较小算子的数量。输出中每个有效位置的索引、短序列尾部和 padding 都应符合原语义。需要额外进行真实模型的输出/精度回归。

## 本地检查边界

Windows 本地已执行 124 组实际 kernel 函数体的 CPU 数值模拟，验证索引计算；这不是 Triton 编译或 NPU 执行。Python 语法、Ruff 和 diff 格式检查另行记录。仓库 UT 在收集阶段因缺少 vLLM 失败，完整格式脚本因缺少 pre-commit 未运行完成。无本地 NPU，因此上述设备测试、图回放和性能测量均待实验环境执行。
