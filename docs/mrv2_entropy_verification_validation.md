# MRV2 Entropy Verify 验证方案

## 范围与状态

本方案验证 MRV2 熵校验的正确性、模型精度、推理性能和草稿采信率。实现复用主线已有的 NPU `rejection_sample` 入口和上游采样器流程，不新增采样器类或类方法 patch。

本次只迁移 `enable_entropy_verify`。主线已有的 `rejection_sample_method="block"` 保留原行为；Ascend `enable_block_verify` 继续使用 MRV1。熵校验与上游 block 模式互斥，synthetic 模式使用配置的采信率，greedy 请求使用原有贪心判断。

**当前无 NPU 环境，所有 NPU 算子、模型、精度和性能验证均待执行。** CPU 参考计算和静态检查不能代替目标硬件验证。

## 对照组与固定条件

| 组别 | 配置 | 用途 |
| --- | --- | --- |
| T | 关闭 speculative decoding | 模型质量基线 |
| S | MRV2，standard，熵校验关闭 | 采样正确性、性能和采信率基线 |
| E | MRV2，standard，熵校验开启 | 本次特性 |
| U | MRV2，upstream block，熵校验关闭 | 主线 block 功能回归 |

对 S/E 使用相同的目标模型、训练好的 draft/MTP 权重、tokenizer、dtype、TP/DP、设备、数据集、随机种子、采样参数、输出长度、并发度和调度设置。记录 vLLM/vLLM Ascend 提交号、容器 digest、CANN、驱动、torch/torch_npu、Triton 版本及完整命令。基线分支和新分支使用同一依赖环境。

先确认新分支 S 与主线 S 的有效输出和计数一致，再比较 S/E。随机采样的 T/S 输出不要求逐 token 一致。E 改变接受规则，不能以逐 token 对齐 T/S 作为精度验收标准。

## 开启与参数矩阵

MRV2 默认开启，可显式设置以记录实验条件：

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
vllm serve /path/to/target-model \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"standard"}' \
  --additional-config '{"rejection_sampler_config":{"enable_entropy_verify":true,"posterior_threshold":0.95,"posterior_alpha":0.4}}'
```

此示例要求模型支持 MTP；EAGLE 等方案保留实际 proposer/model 配置，仅切换熵校验字段。S 设置 `enable_entropy_verify=false`，T 删除 speculative 配置，U 使用 `rejection_sample_method="block"` 并关闭熵校验。

| 维度 | 覆盖值 |
| --- | --- |
| 阈值 / alpha | `(1, 0)`、`(0.6, 0)`、`(0.95, 0.4)`、`(0.95, 1.2)` |
| temperature | `0`、`0.7`、`1.0` |
| logits 处理 | 无截断；top-k/top-p；repetition/frequency/presence penalties |
| logprobs | raw、processed；关闭和开启返回 |
| 执行方式 | eager、项目支持的图模式；异步调度 |
| 草稿 | one-hot、完整 draft logits；步数 `1/2/3/5` |
| 输入形状 | 单请求、多请求、不同长度、bonus only、placeholder、分块 |
| dtype / 词表 | FP32/FP16/BF16；小词表、8192 边界两侧、真实词表 |

先执行默认参数的完整回归，再做参数扫描。`(1, 0)` 恢复单位阈值；浮点边界处需考虑熵规则使用 `>=`、标准路径使用 `>` 的区别。

## 数值正确性与回归

对于经过 temperature、top-k/top-p 和 penalties 处理后的目标分布 `p`，逐位置计算：

```text
H_i = -sum(p_i * log(p_i + 1e-10))
t_i = min(exp(-H_i * posterior_alpha), posterior_threshold)
accept_i = p_i(draft_i) / q_i(draft_i) >= t_i * u_i
```

one-hot 草稿的 `q_i(draft_i)=1`。使用与 NPU 核函数相同的随机数，独立 FP32 oracle 比较接受前缀长度及已接受 token。首次拒绝终止前缀；placeholder 和零概率 draft 是拒绝边界。重采样继续使用标准 residual `max(p-q, 0)`，bonus 使用目标分布。

准备好的自动化覆盖：

| 测试文件 | 验证内容 |
| --- | --- |
| `tests/ut/worker/v2/test_rejection_verify_reference.py` | 阈值改变接受结果、逐位置阈值、首拒绝、placeholder、零概率、alpha=0 |
| `tests/ut/worker/v2/test_entropy_verify.py` | worker 配置、显式覆盖、配置重建隔离、processed logits 传递、模式互斥、非法参数、关闭/synthetic 跳过熵核函数 |
| `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample_v2.py` | FP32 oracle、多个 dtype/词表、参数敏感性、placeholder/bonus、greedy、synthetic、混合请求和分块一致性；保留主线 block 用例 |
| `tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_resample.py` | 熵校验拒绝后的 residual 分布；保留原有恢复采样测试 |
| `tests/e2e/pull_request/one_card/model_runner_v2/test_deepseek_mtp.py` | E/S 配置、图/eager、raw/processed logprobs、多请求、多轮状态复用 |

在已配置依赖的 Linux 环境运行，NPU 测试在目标硬件执行：

```bash
python -m pytest -q tests/ut/worker/v2/test_entropy_verify.py tests/ut/worker/v2/test_rejection_verify_reference.py
python -m pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rejection_sample_v2.py
python -m pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_resample.py
python -m pytest -q tests/e2e/pull_request/one_card/model_runner_v2/test_deepseek_mtp.py -k entropy_verify
```

模型集成测试可能使用随机初始化的 MTP 权重，只能证明流程可运行。精度、采信率和加速收益必须使用匹配的训练权重另行测量。

补充手工压力验证：长时间运行、取消请求与槽位复用、接近上下文长度限制、不同词表大小的目标/草稿、实际模型的采样参数组合。保留失败输入、seed、原始 logits、接受计数与设备报错。

## 模型精度

使用固定版本的数据集和评测器，至少包括确定答案任务（数学/知识）、代码执行任务和业务代表性文本。固定 prompt 模板、样本顺序、最大生成长度和停止条件。先用 T/S 确认基础 speculative 路径质量正常，再比较 E。

随机任务至少使用 3 个 seed；记录每样本输出和评分，以配对差异及置信区间评估波动。按任务分别报告 accuracy、pass@1 或业务评分，避免只给一个总平均数。性能测试可以固定输出长度；精度评测保持正常停止条件。

验收阈值在实验前确定，例如关键任务平均分下降不超过约定的绝对百分点，且单项业务指标不越过底线。不要将采信率提升作为精度通过的证据。

## 性能

对 S/E 使用相同负载，先预热完成编译和图捕获，再开始计时。每个组合至少测量 3 次，交替执行 S/E，避免温度、缓存和后台负载造成固定顺序偏差。

覆盖低并发延迟和高并发吞吐、短/长输入与短/长输出。记录输出 token/s、request/s、端到端时延、TTFT、TPOT/ITL 的 P50/P95/P99、峰值显存及错误率。统计实际输出 token 数，确保不同组的工作量可比。

算子基准使用预分配输入、预热和计时区间前后的 NPU 同步，分别记录熵统计、拒绝判断、resample 和整个 verifier 耗时。端到端 benchmark 保持正常异步执行；profiler 另开短跑，不纳入正式性能统计。

熵校验增加词表扫描和中间张量，只有新增接受 token 带来的收益超过这部分开销才会加速。关注关闭熵校验的 S 相比主线是否出现回归，以及大词表、低采信率负载下 E 的开销。

## 采信率

在测量窗口开始和结束保存 `/metrics`，对计数器取增量，剔除预热流量；多 DP 实例先累加计数再计算比率。进程重启或计数器 reset 的窗口作废。字段以运行版本实际暴露的名称为准，常用项为：

- `vllm:spec_decode_num_drafts`：draft 轮数 `R`。
- `vllm:spec_decode_num_draft_tokens`：提议 token 总数 `D`。
- `vllm:spec_decode_num_accepted_tokens`：接受 draft token 总数 `A`。

报告 `A/D`（token 采信率）、`A/R`（每轮平均接受草稿数）和各位置的接受情况。不要把 bonus/recovery token 算入 `A`；变长请求不能用配置的最大步数代替实际 `D`。分母为零时记为 N/A。

如采集了逐轮数据，位置 `i` 的无条件采信率为该位置接受次数除以实际提议次数；条件采信率以到达该位置的实际验证次数为分母，须单独注明。参数对照使用同一数据集；真实生成前缀会变化，端到端采信率不要求严格单调。固定 logits 的算子测试负责验证阈值规则。

## 结果记录与验收

每组保存配置 JSON、版本、日志、pytest/JUnit、原始输出、逐样本评分、benchmark JSON、metrics 前后快照和必要的 profiler 结果。推荐记录表：

| 组别/参数 | 数据集/负载 | 精度及区间 | 输出 token/s | TTFT P95 | TPOT P95 | A/D | A/R | 显存峰值 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S | 待填 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 未执行 |
| E | 待填 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 未执行 |

验收要求：数值 oracle 与边界回归通过，默认关闭路径和主线 block 功能无回归，模型质量达到预先约定标准，性能与采信率结果可复现。分开标记“已通过”“失败”“未执行”；在 NPU 验证完成前不宣称精度、性能或采信率提升。
