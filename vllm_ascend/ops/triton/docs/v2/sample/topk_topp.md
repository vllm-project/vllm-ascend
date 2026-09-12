# _topk_topp_kernel / apply_top_k_top_p_triton

Source: `vllm_ascend/ops/triton/v2/sample/topk_topp.py` (host wrapper: `apply_top_k_top_p_triton`).

## 功能说明 (Description)

- **算子功能**: 基于 [Qrita](https://arxiv.org/abs/2602.01518)（Pivot-based Truncation and Selection）算法，用单个 Triton kernel 完成 top-k 与 top-p 采样掩码的融合计算：先按 logit 值做 top-k 截断，再对剩余 token 的 softmax 概率做 top-p 截断，被过滤位置写 `mask_value`（默认 `-inf`）。替代 v1 采样路径 `_apply_top_k_top_p_pytorch` 的 `softmax → sort(全词表) → cumsum → masked_fill` 链（7-9 个 kernel launch、每行 O(V·logV) 排序、~1ms/步），并替代 MRV2 路径 `worker/v2/sample/apply_top_k_top_p.py` 对朴素实现的回落。

- **计算公式**（逐行独立处理，`[batch, vocab]` fp32 logits）:

    - Top-k（按 logit 值保留最大的 $k$ 个，边界重复值按数量截断）:
      $$
      mask_i^{(k)} = \mathbb{1}[logit_i > pivot_k]
      $$
      其中 $pivot_k$ 通过对候选区间的三元搜索（ternary search）求得，使严格大于 $pivot_k$ 的元素数 $\ge k$ 且去掉最小边界组后 $< k$。

    - Top-p（在 top-k 保留集合上按概率截断，保留累积概率首次达到 $p$ 的最小集合）:
      $$
      prob_i = \frac{e^{logit_i - max}}{\sum_{j \in keep_k} e^{logit_j - max}}, \qquad
      mask_i^{(p)} = \mathbb{1}[prob_i > pivot_p]
      $$

- **算法流程**（逐行独立处理；Top-k 与 Top-p 可单独或组合启用）:

    1. **第 0 遍（采样统计）**: 从首个 `BLOCK_SIZE` 采样块估计均值/标准差（排除 `-inf`，兼容 grammar bitmask 掩码后的 logits），结合查询表（percentile→σ / 正态 CDF→σ）得到 Gaussian sigma 截断的 outlier pivot。
    2. **第 1 遍（候选聚集）**: 扫描全词表，计算 max/min/有限值计数，将高于 outlier pivot 的候选 logit **按原位置稠密**写入 BUFFER（非候选位置写 `-inf`），并统计候选数 `num_outliers`。
    3. **第 2 遍（top-k 三元搜索）**: 对候选区间（或候选不足时的全量区间）做三元搜索求 $pivot_k$，单次融合扫描同时统计两个探针 pivot 的 `k_pivots_num`、`min_larger`、`num_min_larger`（见 `_update_min_larger_stats` 的 tile 级合并规则），至多 18 次迭代。
    4. **第 3-4 遍（top-k 后的 top-p）**: 对 top-k 保留集合做 softmax（`exp(-inf)==0` 处理被拒条目），将概率按原位置稠密写回 BUFFER；候选不足时回退为对原始 logits 全量 softmax。
    5. **第 5 遍（top-p 二分搜索）**: 对概率区间二分搜索 $pivot_p$，终止时 pivot 与边界统计量来自同一次评估（强制终止路径同样保持一致，避免 `final_pivot` 与 `duplicate_logit` 不一致导致的边界精度问题）。
    6. **第 6 遍（应用掩码）**: 按最终 pivot 生成 keep_mask，含边界重复 logit 的数量控制（`tl.cumsum` 前缀配额，低索引优先），被过滤位置写 `mask_value`。
    7. 仅 top-p（无 top-k）时走独立分支：sigma 截断候选 + 概率二分搜索，若候选质量 $\le p$ 则回退为全量概率搜索。

- **支持模式**: Atlas A2、Atlas A3、Ascend 950（A5）。v1 路径中 A2/A3 非 reduce-sample 模式走本 kernel；A5 与 reduce-sample 模式经 `_apply_top_k_top_p_ascend` 回退到原 `_apply_top_k_top_p_torch_npu` 路径（A5 的 capability 表不含 `NPU_TOP_K_TOP_P`，其非 reduce-sample 模式实际是 pytorch sort+mask，与本 kernel 是两个实现）；MRV2 路径（`patch_triton.py` → `apply_top_k_top_p_npu`）全部硬件走本 kernel（无 Triton 时回退 pytorch）。

## 参数说明 (Parameters)

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:--------|:----------------|:------|:---------|:---------|
| `logits` | 输入/输出 | 二维 `[batch_size, vocab_size]` 采样 logits；可能被原地修改并返回（最后一维非连续时先 `.contiguous()` 返回新张量） | fp32 | ND |
| `LOGITS_STRIDE_0` | 属性 | `logits.stride(0)` 行跨度 | int | 标量 |
| `BUFFER` | 输入/输出（内部） | 每个向量核一行的候选/概率暂存区 `[num_programs, vocab_size]`，host 侧按 `(device, dtype, vocab)` 缓存复用 | fp32 | ND |
| `PERCENTILE_TO_STD_TABLE` | 输入（常量） | top-k percentile→σ 查询表（200 项，host 侧按 device 缓存） | fp32 | ND |
| `NORMAL_CDF_TO_SIGMA_TABLE` | 输入（常量） | top-p 正态 CDF→σ 查询表（200 项） | fp32 | ND |
| `K` | 输入 | 一维 `[batch_size]` 每行 top-k 保留数；`None` 禁用 top-k；`k >= vocab_size` 视为禁用 | int32 | ND |
| `P` | 输入 | 一维 `[batch_size]` top-p 值，范围 `[0, 1]`；`None` 禁用 top-p；`p == 1.0` 视为禁用 | fp32 | ND |
| `MASK_VALUE` | 属性 | 被过滤位置填充值，默认 `-inf` | constexpr float | - |
| `VOCAB_SIZE` | 属性 | 词表大小（编译期常量，决定 tile 数） | constexpr int | - |
| `BLOCK_SIZE` | 属性 | 扫描 tile 宽度；NPU 取 4096（见"约束说明"），CPU 取 256 | constexpr int | - |
| `BLOCK_SIZE_TRUNC` | 属性 | 搜索阶段 tile 宽度；NPU 取 2048，CPU 取 128 | constexpr int | - |
| `TOPK_ENABLED` / `TOPP_ENABLED` | 属性 | 编译期开关，组合出仅 top-k / 仅 top-p / 融合三种特化 | constexpr bool | - |

## 约束说明 (Constraints)

- `logits` 必须为二维 fp32；行内 vocab 维度必须连续（`stride(1) == 1`），行跨度任意。
- `k >= vocab_size` 的行是 no-op（kernel 内 `if k < VOCAB_SIZE` 跳过）；`p == 1.0` 的行跳过 top-p；`batch_size == 0` 或 `k`、`p` 同时为 `None` 时直接返回输入。
- 大量 `-inf` logits（grammar/structured-output bitmask 场景）不会产生 NaN：统计与搜索显式排除 `-inf`；有限值数 $< k$ 时保留全部有限值；全 `-inf` 行为 no-op。
- 边界重复值按"低索引优先"的配额保留（`tl.cumsum` + `duplicate_count <= num_keep`）；top-p 强制终止路径的 pivot 与边界统计量取自同一次评估。**tie 语义注意**：当 top-k/top-p 截断边界恰好落在精确 tie 组内时，本 kernel 按配额截断到恰好 k/p 个（低索引优先），而 pytorch sort+截止路径会整组膨胀保留——这是两种实现的固有语义差异（连续 randn fp32 下边界精确 tie 概率为 0，随机数据对拍不受影响）。因此"逐位相等"承诺仅在边界无精确 tie 时成立；对拍测试对含 tie 场景断言 `<=` 方向。
- **BLOCK_SIZE=4096 / BLOCK_SIZE_TRUNC=2048 的取值依据**: A2/A3 统一缓冲区（UB）为 192 KB，搜索循环同时保有两个探针 pivot 的活跃 tile 变体与稠密临时量，4096-wide fp32 tile（16 KB）留足裕量；遵循树内 penalties/min_p 等 sampler kernel 的 tile 先例。此前移植（PR #13847）取 1024 导致 V=128K 需 126 个 tile 迭代、性能全面劣化，本取值将其降为 32/16，是 A3 review 给出的直接修复。
- 每个 program 独占 BUFFER 的一行（`BUFFER + pid * VOCAB_SIZE`），launch grid 为 `min(num_vectorcore, batch_size)`，program 内 grid-stride 循环处理多行——同一行的两次访问（不同 pass）之间没有跨 program 依赖。
- `multibuffer=False` 启动（与 grammar bitmask kernel 相同，关闭多缓冲以更好利用 UB）。
- 图模式：支持。grid 仅依赖设备向量核数与 host 侧 shape。
- triton-ascend 平台适配（与上游 GPU 版的结构性差异）：
    - 候选聚集不做运行时索引 scatter（上游 `BUFFER_ROW + write_pos`，`write_pos` 由 `tl.cumsum` 派生），改写为按原位置稠密存 + `-inf`/0.0 填充——scatter 在 triton-ascend 上会 lower 为 DiscreteMemAccess/SyncBlockLock 使 kernel 串行化。代价是各搜索 pass 需扫描全词表宽度（-inf/0.0 lane 被值掩码跳过），收益是全部访存保持连续。
    - 标量计数器一律 int32（上游部分用 uint32，triton-ascend 标量 uint32 累加 lower 受限）。
    - 查询表用 masked 向量 load + `tl.sum` 归约（triton-ascend 不支持标量索引的 `tl.load`）。

## 与上游实现的差异 (Origin and Differences)

- **Origin**: 移植自 vLLM `vllm/v1/sample/ops/topk_topp_triton.py`（Qrita 算法，含上游对 grammar `-inf` logits 的鲁棒性处理与强制终止一致性修复）。
- **Differences**:
    - NPU 性能适配：launch grid 从 `num_compute_units`（NPU 上语义为 Cube Core 数）改为 `min(get_vectorcore_num(), batch_size)` 向量核网格 + grid-stride 行循环；`BLOCK_SIZE` 8192→4096、`BLOCK_SIZE_TRUNC` 4096→2048（A2/A3 UB 192 KB）；`multibuffer=False`。
    - triton-ascend lower 限制适配：稠密 BUFFER 替代运行时索引 scatter；int32 标量计数；向量化查询表。
    - 针对 vllm-ascend 逻辑的修改：top-p 强制终止时 pivot 与边界统计量同源（PR #13847 在 A3 上暴露的 top-p 边界精度问题的修复）；`num_keep` 下限钳到 1（至少保留一个 token）；MRV2 入口 `apply_top_k_top_p_npu` 在 k/p 均 None 时以 k=V/p=1.0 跑一次 kernel（防御性兜底：若未来有调用方在 warmup 期到达该 hook，可顺带完成 BUFFER 分配）。

## 测试用例 (Test Cases)

> [!NOTE]
> 单算子精度测试位于 `tests/e2e/nightly/single_node/ops/singlecard_ops/triton`。

`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_topk_topp.py` 对拍 vllm-ascend 的 sort+mask pytorch 参考实现（`_apply_top_k_top_p_pytorch`）：top-k only（逐位相等）、top-p only、top-k+top-p（保留位置一致 + 数量容差）、mixed-k batch、k=None/p=None no-op、k==V no-op、k=1/k=V/混合极值、p 极值、非连续输入、`-inf`/grammar 掩码、等值边界（tie 场景按"约束说明"所述语义断言 `<=` 方向）、大 batch。形状取 NPU 服务模型词表（1024/32000/128256）。

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_topk_topp.py
```

CPU 可跑的调度逻辑 UT 见 `tests/ut/sample/test_topk_topp_triton.py`（dispatch 守卫矩阵、reduce-sample/A5 回退、warmup 分支、pytorch fallback）。
