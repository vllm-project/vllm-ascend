# Token/Logprob 异常检测与动态 Dump 方案

## 1. 目标

在现有投机接受率异常检测之外，增加基于 **输出 token + top-k logprobs** 的在线异常检测（生僻字 / 乱码 / 重复 / NaN），与接受率检测共用动态 msprobe dump 与 `debug_log_full` 通路。

设计原则：

- **尽快检出**：小窗口、低命中阈值；重复略严一点防误报。
- **滑窗外置**：Dumper 维护队列；detector 配成「一次调用 = 一窗」，避免双重滑窗。
- **可配置开关**：spec acceptance / token_logprob 分别使能。
- **复用 dump**：命中后走 `enable_msprobe_dump_if_needed`。

## 2. 开关与配置

通过 `additional_config.dynamic_dump_config`：

| 字段 | 默认 | 说明 |
|------|------|------|
| `enable_spec_acceptance_check` | `true` | 投机接受率检测（兼容旧行为） |
| `enable_token_logprob_check` | `false` | token/logprob 检测（需请求开启 logprobs） |
| `token_logprob_window` | `64` | 每请求缓冲长度 = 送检窗长 |
| `token_logprob_stride` | `32` | 满窗后每新增 N token 再检 |
| `token_logprob_topk` | `20` | 每位置最多保留 top20 |
| `ill_nan_window_thresh` | `1` | NaN/Inf 命中窗数 |
| `ill_rare_window_thresh` | `1` | 生僻字 |
| `ill_garbled_window_thresh` | `1` | 乱码 |
| `ill_repet_window_thresh` | `2` | 重复（半重叠两窗确认） |
| `dynamic_dump_max_times` | `0` | `0` 时不触发 dump（检测也会直接 return） |

示例：

```json
{
  "dynamic_dump_config": {
    "enable_spec_acceptance_check": true,
    "enable_token_logprob_check": true,
    "dynamic_dump_max_times": 3,
    "token_logprob_window": 64,
    "token_logprob_stride": 32,
    "token_logprob_topk": 20,
    "ill_repet_window_thresh": 2
  }
}
```

请求侧需设置 `logprobs >= 1`（建议 `>= 20`），否则 worker 无 topk 数据，检测跳过。

## 3. 架构

```text
model_runner_v1 (sample/bookkeeping 后)
  ├─ clear_finished_requests(...)   # 唯一清理由处（spec history + token 缓冲）
  ├─ check_all_spec_acceptance(...) # 内部 enable / max_times 门控
  └─ check_all_token_logprobs(...)  # 内部 enable / max_times 门控；不再 clear
        │
        ▼
Dumper 每请求 deque(maxlen=window)
  满窗 / 之后每 stride 新 token
        │
        ▼
msprobe ILLDetector.detector(topk_dicts, tokens, model_config)
  （内部 window=stride=队列长 → 单窗）
        │
        ▼
按 ill_type 累加命中次数，达 thresh → enable_msprobe_dump + debug_log_full
```

### 3.1 为何不两边各滑一套

- msprobe 默认 `window_size=128, stride=64`，且 `single_window_thresh=14` 适合离线长序列。
- 在线：Dumper 队列长度 = 窗长；构造 detector 后覆盖为 `window_size=stride=token_logprob_window`，并把 garbled/repeat 的内部多窗阈值置 0，使 **单次调用能返回 is_ill**。
- **多窗投票改由 Dumper 的 `ill_*_window_thresh` 完成**，便于尽快检出且可启动配置。

### 3.2 logprobs 布局

vLLM `LogprobsLists` 每行：`[sampled_logprob, top1, …, topk]`。

Dumper `_row_to_topk_dict`：按 logprob 降序取前 `token_logprob_topk`，转成 `Dict[token_id, logprob]` 再交给 detector（detector 内部会再排序截断）。

MTP / 投机：一步多个 accepted token → 多行 logprobs，按序 append；使用 `cu_num_generated_tokens` 切片。

### 3.3 model_config / tk2cat

- 传入 `{"model_name": Path(model).name}` 供名称模糊匹配。
- `get_tk2cat` 依赖「末 token 为 eos」校验；生成中途常走 **无词表 top1 阈值** 路径。类别增强需预加载 tk2cat（后续优化）。

## 4. 代码落点

| 模块 | 变更 |
|------|------|
| `vllm_ascend/ascend_config.py` | `DynamicDumpConfig` 新字段与校验 |
| `vllm_ascend/dumper.py` | 缓冲、命中计数、`check_token_logprob_anomaly` / `check_all_token_logprobs`、`clear_finished_requests`；投机接受率受 `enable_spec_acceptance_check` 门控 |
| `vllm_ascend/worker/model_runner_v1.py` | 与接受率检查同位置调用 token_logprob 检查 |
| `docs/.../additional_config.md` | 配置表 |

## 5. 生命周期与资源

- 每请求缓冲：最多 `window × topk` 个 (id, logprob)。
- 请求结束：`clear_finished_requests` 销毁缓冲与命中计数。
- 检测时日志：`active_reqs`（当前缓冲请求数）、`ill_type`、hits。

## 6. 与 dump 共用策略

- 冷却 / 最大次数 / 每请求只 dump 一次：沿用 `enable_msprobe_dump_if_needed`。
- TP0 + PP last 打详细日志；dump 状态写 msprobe 配置文件。
- `debug_log_full` 在 dump 使能成功后置位，snapshot 到 `ModelRunnerOutput`。

## 7. 限制与后续

1. 未开 `logprobs` 时无法做 token_logprob 检测。
2. 中途无 eos → tk2cat 可能不可用。
3. v2 runner 可按同样 API 在 `postprocess_sampled` / `sample_tokens` 后接入。
4. 若需更激进：减小 `window`/`stride`，或将 `ill_repet_window_thresh` 设为 `1`。
