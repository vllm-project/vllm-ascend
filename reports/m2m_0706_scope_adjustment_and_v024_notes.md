# m2m-0706 升级范围调整与 v0.24.0 后续升级说明

## 结论

本 PR 的升级范围应调整为：

- `vllm-main-verified.commit`: 升级到 `ba22152096b2484faa3579624a253d54804d876d`
- `vllm-release-tag.commit`: 保持 `v0.23.0`
- `v0.24.0` 不属于本 PR 的升级范围，应作为后续独立 PR 处理

因此，本 PR 里所有只服务于 `v0.24.0` release tag 的改动应回退或拆出；同时，部分版本判断需要从 `0.24.0` 改成 `0.23.0`，因为当前矩阵要同时支持 `ba221520...` main commit 和 `v0.23.0` tag。

## 本 PR 建议处理清单

| 文件/改动 | 当前问题 | 建议处理 | 原因 |
|---|---|---|---|
| `.github/vllm-release-tag.commit` | 当前从 `v0.23.0` 改成了 `v0.24.0` | 回退为 `v0.23.0` | 新策略明确本 PR 不升级 release tag |
| `tests/e2e/pull_request/two_card/test_gpt_oss_distributed.py` | 新增 `vllm_version_is("0.24.0")` skip | 回退 | `v0.24.0` 不在本 PR 范围；该 skip 是为 v0.24 loader 缺口服务 |
| `vllm_ascend/patch/worker/patch_deepseek_v2.py` | 为 `0.24.0` 增加 `config.hidden_size` 分支 | 回退为 `self.hidden_size` | 该 patch 外层已有 `if not vllm_version_is("0.23.0")`，v0.23 不会进入；`ba221...` main 已有 `self.hidden_size` |
| `.github/workflows/_selected_tests.yaml`、`pr_test.yaml`、`run_selected_tests.sh` | 继续执行所有 E2E 的 CI 行为改动混在 main2main PR 中 | 建议拆出或最终回退 | 这是 CI 诊断能力，不是 main commit 适配本身；可临时保留用于暴露问题，合入前建议单独 PR |
| `vllm_ascend/worker/v2/model_runner.py` | 当前 `if not vllm_version_is("0.24.0")` | 改为 `if not vllm_version_is("0.23.0")`，并更新注释 | `v0.23.0` 没有 `InputBatch.is_padding/prompt_lens`，`ba221...` main 有；当前判断会错误地给 v0.23 传新字段 |
| `vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py` | 当前 `if vllm_version_is("0.24.0")` 使用旧 helper 名 | 改为 `if vllm_version_is("0.23.0")` | v0.23 使用 `_compute_global_lse/_compute_block_stats_kernel`；`ba221...` main 使用 `_compute_global_logsumexp/_compute_local_logits_stats_kernel` |
| `vllm_ascend/patch/worker/patch_cudagraph.py` | 当前 `if vllm_version_is("0.24.0") or ...` | 建议改为 `if vllm_version_is("0.23.0") or ...` | 本 PR tag 是 v0.23；如果要保持 tag 行为不变，应把旧行为保护对象改成 v0.23 |
| `vllm_ascend/worker/v2/spec_decode/dflash/speculator.py` | 当前保留 `v0.24.0` 旧 kernel 分支 | 不建议改成 `0.23.0`；建议删除 0.24 分支，仅保留 `ba221...` main kernel 适配 | v0.23 没有上游 DFlash speculator 文件；该分支只服务 v0.24 tag，不属于本 PR |

## 本 PR 仍建议保留的 main commit 适配

| 文件/改动 | 保留原因 | 上游来源 |
|---|---|---|
| `vllm_ascend/patch/platform/__init__.py` 和 `patch_deepseek_v4_tool_call_parser.py` | `ba221...` main 删除旧 `deepseekv4_tool_parser.py`，需要避免 import 旧模块失败 | vLLM #45877 |
| `tests/ut/patch/platform/test_patch_deepseek_v4_tool_call_parser.py` | 旧 parser 不存在时跳过 UT，避免在 main commit 下 import 失败 | vLLM #45877 |
| `vllm_ascend/patch/worker/patch_qwen3_dflash.py` 的 `hasattr` 判断 | `ba221...` main 有 `_read_mask_embedding`，v0.23 没有；能力判断比版本判断更稳 | vLLM #46104 |
| `vllm_ascend/quantization/modelslim_config.py` 的 `get_cache_scale` | `ba221...` main 的 `laguna_dflash.py` 会调用 `quant_config.get_cache_scale(name)` | vLLM #46853 |
| `vllm_ascend/worker/v2/model_runner.py` 的 kwargs 构造方式 | 同一套代码要兼容 `ba221...` main 的新 `InputBatch` 字段和 v0.23 的旧字段 | vLLM main 输入批协议变化 |
| `vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py` 的 helper import 分流 | v0.23 和 `ba221...` main 的 helper 名不同 | vLLM #46781 |
| `vllm_ascend/worker/v2/spec_decode/dflash/speculator.py` 的 main kernel | `ba221...` main 的 DFlash kernel 协议不同，需要 Ascend 侧同步 | vLLM #46995 |
| `vllm_ascend/patch/worker/patch_cudagraph.py` fallback | Ascend 本地 cudagraph patch 在 `ba221...` main 下可能遇到非整除 batch descriptor，需要保护 | vllm-ascend 本地 patch 边界被 main commit 暴露 |
| `vllm_ascend/worker/model_runner_v1.py` 和 `_310p/model_runner_310p.py` 的 `valid_cudagraph_modes` | dummy capture 需要把期望 cudagraph mode 传给 dispatcher；310P override 需要同步父类签名 | vllm-ascend 本地修复 |
| `vllm_ascend/models/deepseek_v4.py`、`llama_eagle3_vwn.py` 的 `spec_step_idx` 可选参数 | 对 MTP/spec decode 调用协议更兼容；不是 0.24 专属 | vLLM main MTP 模型协议演进 |

## 需要额外检查或修改的点

1. PR 中不应再出现新增的 `vllm_version_is("0.24.0")` 判断。
   可以用以下命令检查：

   ```powershell
   rg -n -F 'vllm_version_is("0.24.0")' .github vllm_ascend tests
   ```

2. `.github/vllm-release-tag.commit` 应保持：

   ```text
   v0.23.0
   ```

3. PR 最终描述需要说明：

   - 本 PR 只升级 main commit 到 `ba22152096b2484faa3579624a253d54804d876d`
   - release tag 仍是 `v0.23.0`
   - `v0.24.0` 相关适配会在后续独立 PR 处理

4. 如果保留 CI continue-on-error 改动，需要在 PR 中说明它是临时诊断能力；如果要保持 PR 范围纯净，建议拆成独立 CI PR。

## 后续 v0.24.0 升级需要处理的点

后续单独升级 `vllm-release-tag.commit` 到 `v0.24.0` 时，至少需要处理以下项目。

| 模块 | v0.24.0 问题 | 建议修复 |
|---|---|---|
| `.github/vllm-release-tag.commit` | tag 从 `v0.23.0` 升到 `v0.24.0` | 单独 PR 修改 tag，并确保 CI 矩阵明确跑 main commit + v0.24.0 |
| `InputBatch` 字段 | v0.24.0 仍没有 `is_padding` 和 `prompt_lens` dataclass 字段；`ba221...` main 有 | 版本判断需要覆盖 v0.23 和 v0.24，或改成能力检测，避免给旧 tag 传新字段 |
| rejection sampler helper | v0.24.0 使用 `_compute_global_lse/_compute_block_stats_kernel`；main 使用 `_compute_global_logsumexp/_compute_local_logits_stats_kernel` | import 分流需要覆盖 `v0.24.0` |
| DeepSeekV2Model hidden_size | v0.24.0 的 `DeepseekV2Model` 没有 `self.hidden_size`；main 后续由 vLLM #46986 修复 | 在 v0.24.0 下使用 `self.config.hidden_size` |
| Qwen3 DFlash `_read_mask_embedding` | v0.24.0 没有 `_read_mask_embedding`；main 有 | 保留 `hasattr(DFlashQwen3ForCausalLM, "_read_mask_embedding")` 能力判断 |
| DFlash speculator kernel | v0.24.0 和 main 的 DFlash kernel 协议不同 | 需要保留 v0.24 旧 kernel 和 main 新 kernel 的分流 |
| GPT-OSS E2E | v0.24.0 缺少后续 GPT-OSS loader 修复，`unsloth/gpt-oss-20b-BF16` 可能失败 | 二选一：跳过 v0.24 GPT-OSS E2E，或 backport vLLM #45818/#46441 相关 loader 修复 |
| DeepSeek V4 tool parser | v0.24.0 仍有旧 `deepseekv4_tool_parser.py`；main 改成 engine parser | 当前 `find_spec` 保护可以同时兼容 v0.24 和 main |
| ModelSlim `get_cache_scale` | v0.24.0 没有 `laguna_dflash.py` 的 `get_cache_scale` 调用；main 有 | 该方法可保留，不影响 v0.24 |
| cudagraph patch | v0.24.0 可能需要保持旧断言行为，main 需要 fallback | 分支条件需要明确保护 v0.24 或改为更稳的能力/场景判断 |

## 建议的后续执行顺序

1. 先按本 PR 新范围清理代码：回退 tag、回退 v0.24-only skip、调整 0.23/main 分流。
2. 本 PR 只跑并修复 `ba221520...` main commit 与 `v0.23.0` tag 矩阵。
3. 当前 PR 合入或稳定后，再开独立 PR 升级 `v0.24.0` tag。
4. v0.24 PR 中按上表逐项适配，并在 PR 描述中明确这些改动是 release tag 升级带来的，不混入 main commit 升级逻辑。
