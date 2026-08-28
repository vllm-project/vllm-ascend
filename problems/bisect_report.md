# Qwen3-235B 32768/4096 性能劣化二分定位报告

日期：2026-08-28　服务器：192.168.13.197（zty_piecewise_precision 服务容器 / zty_aisbench 压测容器）
场景：Qwen3-235B-A22B-w8a8-QuaRot，DP4×TP4（EP），输入 32768 / 输出 4096，60 请求，temperature=0 + ignore_eos
现象：vllm-ascend v0.26.0rc + vllm v0.26.0 输出吞吐 271.9 tok/s（TPOT 48.9ms），基线 v0.23.0 + v0.23.0 为 299.2 tok/s（TPOT 44.3ms）

## 一、结论（TL;DR）

**引入劣化的 PR：#12228 `[Revert][Refactor][Attention] Restore paged attention fallback (reverts #11899)`（vllm-ascend main 76ab5e28b，2026-07-20 合入，随 v0.25.1/v0.26.0rc 发布）。**

机制：该 revert 恢复了 Paged Attention (PA) fallback 及 `pa_shape_list` 配置项。启动脚本里显式配置了
`--additional-config '{"ascend_scheduler_config":{"enabled":false},"pa_shape_list":[4,8,16,32,48,64,96,128,160,192]}'`，
在 `cudagraph_mode=FULL_DECODE_ONLY` 下，decode batch 命中该列表的 shape（本场景 15 并发摊到 4 个 DP rank，batch 以 4 为主）会改走 `torch_npu._npu_paged_attention`，而不是 FlashInferAscend (FIA) 路径。**32k 长上下文下 PA decode 比 FIA 慢约 12%**，导致吞吐从 ~300 掉到 ~260-272。

**修复（已在 v0.26.0rc 验证）：从服务启动配置中删除 `pa_shape_list`**，decode 全走 FIA：
v0.26.0rc + 原配置 = 271.9 tok/s → v0.26.0rc + 删除 pa_shape_list = **306.97 tok/s**（+12.9%，超过 v0.23.0 基线 299.2）。
修复版启动脚本：`/mnt/share/z00586359/run/performance/start_vllm_server_nopa.sh`（其余参数与原脚本完全一致）。

## 二、二分过程与数据

方法：vllm-ascend 每个 commit 通过 `.github/vllm-release-tag.commit` 锁定配套 vllm 版本，两点联动切换、双仓重建、重启服务、ais_bench 压测。自动化脚本 `bisect/point.sh`，结果记录 `bisect/results.tsv`，日志 `bisect/logs/<label>/`。

二分阶梯（vllm-ascend main 血统，good→bad 阈值 285 tok/s）：

| 时间 | vllm-ascend | vllm | pa 生效? | 输出吞吐 tok/s | 判定 |
|------|-------------|------|----------|----------------|------|
| (任务给定) releases/v0.23.0 | 5cb98caaa | v0.23.0 | 是 | 299.238 | good 基线 |
| 03:54 | ddc85dda76 (v0.25.1分支点) | v0.25.1 | 是 | 264.5 | bad |
| 04:44 | 239c64b0a9 (v0.24.0分支点) | v0.24.0 | 否(PA已被#11899移除) | 295.6 | good |
| 05:12 | fe7bfc474 (pin→v0.25.0) | v0.25.0 | 否 | 306.7 | good |
| 05:45 | 95ef5af64 (pin→v0.25.1) | v0.25.1 | 是 | 261.5 | bad |
| 06:21 | 76ab5e28b (+#12228 revert) | v0.25.0 | **是** | 258.3 | **bad ← 边界** |
| 07:06/08:02/11:14 | fd69c96a5 (#12017) | v0.25.0 | 否 | 281.8 / 283.9 / 293.1 | 漂移噪声，见四 |
| 07:33/09:03 | 0f9fc6850 (#12420) | v0.25.0 | 否 | 306.3 / 291.3 | good |
| 08:33 | fd69c96a5 文件级回退 | v0.25.0 | 否 | 292.2 | ≈同窗口good |
| 09:34 | 76ab5e28b + 去掉pa_shape_list | v0.25.0 | 否 | **293.6** | **good ← 机制验证** |
| 10:22 | cf0baa38d (v0.26.0rc HEAD) + 去掉pa_shape_list | v0.26.0 | 否 | **306.97** | **good ← 修复验证** |
| (任务给定) cf0baa38d 原配置 | v0.26.0 | 是 | 271.9 | bad（劣化端点） |

关键判定链：
1. fe7bfc474(v0.25.0) good=306.7 与 95ef5af64(v0.25.1) bad=261.5 之间只有 3 个代码提交（0f9fc6850 / fd69c96a5 / 76ab5e28b）+ vllm v0.25.0→v0.25.1 跳变（vllm 侧仅 2 个 bugfix 提交，且只影响 FP8/NVFP4 pattern，与 ascend int8 w8a8 无关）。
2. 76ab5e28b 在 pin 仍为 v0.25.0 时即 bad(258.3) → **vllm 侧排除，真凶是 76ab5e28b**。
3. 同一 commit 76ab5e28b 仅删掉 pa_shape_list → 293.6（+35）→ **机制 = PA 被 pa_shape_list 重新启用**。
4. v0.26.0rc HEAD 删 pa_shape_list → 306.97 → **修复有效且反超基线**。

## 三、为什么基线 v0.23.0 同样配了 pa_shape_list 却不劣化

v0.23.0 的 `using_paged_attention()` 判定与现在逐字节相同（FULL_DECODE_ONLY + shape 命中即走 PA），基线也是 PA/FIA 混跑，说明**当时 PA≈FIA**（旁证：v0.24.0 全 FIA = 295.6 ≈ v0.23.0 混跑 299.2）。v0.24→v0.25 期间 FIA decode 路径明显变快（v0.25.0 全 FIA = 306.7，v0.26.0rc 全 FIA = 307），PA 未同步提升。#12228 把 PA fallback 连同 pa_shape_list 语义恢复后，老配置在新时代选中了慢路径，形成"相对劣化"。

PA 的 per-step 开销点（供后续优化参考）：`update_graph_params()` 中对全部 94 层逐层执行 `_npu_paged_attention_get_workspace` + `graph_task_update_begin/end` + op 重提交 + event record，长上下文 GQA 下不如 FIA 图回放高效。

## 四、fd69c96a5 (#12017 MRV2 DSpark FullGraph) 的排查结论

晨间数据显示 fd69c96a5 = 281.8/283.9（对比同时段 good 306），疑似 -8%。但：
- 该提交只改 3 个文件，全部在 V2 model runner / DSpark spec-decode 路径（本场景 V1 + 无 spec decode 不经过）；
- 文件级回退实验（把该 2 个代码文件回退后重测）= 292.2，与同窗口 good 参照 291.3（0f9fc6850 复测）无差异；
- 同一 commit 0f9fc6850 两次测量 306.3 vs 291.3，证明存在 ±3% 左右的跨时段环境漂移（共享宿主机）。
结论：fd69c96a5 的表面劣化在漂移范围内、无作用机制，**不判定为劣化源**。同窗口复测（two_commits_r3，11:14）= 293.1 tok/s，与同窗口 good 参照（one_commit_r2=291.3 / files_revert=292.2 / revert_nopa=293.6）完全一致，确认洗清。

## 五、修复与建议

1. **部署侧（已验证）**：`start_vllm_server.sh` 的 additional-config 删除 `"pa_shape_list":[...]`（等价使用 `start_vllm_server_nopa.sh`）。v0.26.0rc 吞吐 271.9→306.97 tok/s，TPOT 相应回落，恢复并超过基线。注意：frozen 的历史结论 271.9/299.2 都是在带 pa_shape_list 的配置下测得的，后续对齐数据时统一采用新配置。
2. **上游建议（vllm-ascend）**：
   - `pa_shape_list` 恢复后默认语义变强（老配置静默切换后端），建议在文档/release note 标注 A3 GQA 长上下文模型（如 Qwen3-235B）decode 应优先 FIA、pa_shape_list 仅用于 PA 已知更优或 FIA 不支持的 case（如 Gemma4 512 head）；
   - 或考虑 `using_paged_attention()` 对 GQA + 长上下文默认返回 False，把 PA fallback 收窄为显式 opt-in；
   - PA 路径 per-layer graph_task_update 提交流程有优化空间。

## 六、可复现实验资产

- 二分自动化：`/mnt/share/z00586359/run/performance/bisect/point.sh <sha> <label> [start_script]`（EXTRA_CMD 支持文件级回退实验）
- 全部结果：`bisect/results.tsv`；每轮 serve/bench/构建日志：`bisect/logs/<label>/`
- 修复版启动脚本：`/mnt/share/z00586359/run/performance/start_vllm_server_nopa.sh`
