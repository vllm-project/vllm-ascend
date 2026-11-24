# M3.1 BF16 基座重载修复验证

2026-11-25，A3 / Ascend 910_9382，真实 4 层权重，BF16/eager。
**LoRA 开启后的基座 checkpoint 重载问题已修复，7 组独立冷启动对照通过。**
配套 vLLM 保持 `c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15`，未修改上游仓库；
本次 Ascend 修改基于个人分支的 `a1cdf0b3217322e50e7674c9a1791f98ed81d83c`。
可复跑命令见 [BASE_WEIGHT_RELOAD.md](BASE_WEIGHT_RELOAD.md)，
逐 rank 比较、原始报告和源码 SHA256 见 [validation-reload-summary.json](validation-reload-summary.json)。

## 根因与修改

LoRA 包装后，参数名增加 `.base_layer.`，checkpoint 键仍保持原名。
M3 手写 loader 直接查 `named_parameters()`，导致投影权重被跳过；
上游专家映射还会因为检测到任意 LoRA 包装而给两侧名称增加 `base_layer.`，
所以只包装 attention 时也会影响未包装的专家。
layerwise reload 对没有加载到新值的层恢复旧张量，因此 RPC 可以返回成功但权重没有更新。

生产修改仅位于 `vllm_ascend/models/minimax_m3/minimax_m3.py`：

- `:1295` 的 `get_expert_mapping` 将专家映射两侧规范为 checkpoint 名称。
- `:1327` 的 `load_weights` 保留实际参数名，使用 `setdefault` 增加去掉 `.base_layer.` 的别名。

方案与已读取的上游参考提交
[`4917985`](https://github.com/vllm-project/vllm/commit/4917985161655ab0ca549f8b6e07ead88338d078)
一致。继续使用配套的参数 loader、TP/EP 分片、后处理和 layerwise 生命周期，没有新增算子或运行配置。
`tests/ut/models/minimax_m3/test_weight_reload.py:85` 使用真实 LoRA 基类与上游 reload 生命周期，
验证未包装、只包装 attention、完整包装三种情况连续两次更新、全部参数值及原存储地址保留。
先运行未修改 loader 得到 **2 failed / 1 passed**，修复后 **3 passed**；没有删除失败断言或放宽比较。

## 实机结果

独立新 checkpoint 改变 414 个投影和 1 个 norm 对照，融合后每 rank 的 41 个运行时参数中应改变 19 个。
使用真实 `reload_weights(weights_path=..., is_checkpoint_format=True)` 入口；重载后再恢复原 checkpoint。
固定 KV cache 为 512 MiB，为重载临时张量预留空间；prefix caching 与 chunked prefill 关闭。
3/17/129 token prompt 分别 greedy 生成 8 token，检查完整参数指纹、原始 token ID 和 top-5 logprobs。

| 场景 | 全部参数对冷启动 | token / logprobs 对照 | 恢复原基座、adapter 缓冲区 | 结果 |
| --- | --- | --- | --- | --- |
| TP1，LoRA 关闭 | 41/41 | 3/3 | 一致 | PASS |
| TP1，LoRA 开启，无 adapter | 41/41 | 3/3 | 一致 | PASS |
| TP1，base/A/B | 41/41 | 9/9 | 一致 | PASS |
| TP2，base/A/B | 82/82 | 9/9 | 一致 | PASS |
| TP2+EP2，base/A/B | 82/82 | 9/9 | 一致 | PASS |
| 文本入口，TP1，base/A/B | 41/41 | 9/9 | 一致 | PASS |
| HTTP TP2+DP2+EP4，base/A/B | 164/164，含四个 rank | 9/9 | 一致 | PASS |

- 相关 CPU 回归：**1519 passed / 1 skipped**，包括新增 3 个重载回归。
- 仓库完整 `bash format.sh ci` 通过，包含 Markdown、格式和 staged secret scan。
- 全语言模块 BF16 数值回归：**15/15**，覆盖 indexer、普通 MLP、共享专家、路由专家和 all 的 zero/A/B；
  加载切片和非目标切片精确一致，独立 BF16 参考超差元素为 0，容差仍为 `atol=rtol=1/64`。
- 恢复后的四卡 HTTP 服务：真实 base/A/B 同批 **36/36** token 一致；
  base/B/C 槽位复用 **12/12** 一致，A 恢复及 API 卸载/重载一致。
- 通用 checkpoint 准备脚本实跑完成：415 个张量改变；5 个生成 shard 与最初独立资产的完整文件 SHA256 一致。

## 反证与边界复核

修复前的 [失败统计](validation-reload-before-summary.json) 保留：开启 LoRA 时 18/41 参数仍为旧值，
仅 norm 对照更新；无 adapter 时 token 恰好相同，也被参数和 logprob 检查判失败。
这排除了“仅 adapter 缓存错误”“只是 BF16 舍入”及“新 checkpoint 路径无效”的解释。
此前默认 KV 预算下普通模型重载 OOM 的记录也保留；明确 KV 预算后普通模型对照通过。

| 视角 | 位置、证据与适用边界 |
| --- | --- |
| 安全 | `minimax_m3.py:1327` 仅给现有参数对象增加名称别名，仍调用原有 loader；没有新增外部接口。测试开发 RPC 仅在 loopback 服务启用，未写入凭据或权重。 |
| 性能/资源 | `minimax_m3.py:1328` 在加载阶段遍历已建立的参数字典，本例每 rank 41 项；不进入 forward 热路径。重载仍需临时内存，512 MiB KV 是测试资源预算，不是性能结论。 |
| 边界 | `test_weight_reload.py:85` 覆盖 plain、attention-only、all；实机覆盖空 adapter、indexer、MLP 和专家。`setdefault` 保留已存在的规范键，未扩大为其他模型的处理。 |
| 时序 | UT 连续两次更新并检查存储地址；实机逐 rank 检查新基座及恢复、adapter 缓冲区不变，再执行混批和槽位复用。只验证请求完成后的重载。 |

初次 HTTP 采集严格要求四个 rank，但配套 API 只返回第一个 DP engine 的两份结果，因此采集器主动失败。
现于 `reload_observation.py:52` 使用已有 DP CPU group 汇总只读指纹并携带 DP/TP 身份；
`validate_bf16_reload.py:165` 仍要求 `TP×DP` 份快照，没有将两份结果当作四卡通过。
原始诊断、JSON 和日志保留在测试机的
`/workspace/lora-validation/m31-full-bf16-20261124/reload-20261124-001`。

本轮仅验证上述真实减层模型和配置；60 层完整模型、A5、ACLGraph、kernel-format 重载、
请求进行中的并发热更新、PP/PCP 和性能未验证。
配套 vLLM 明确拒绝 direct reload 与 LoRA 共用，本轮未绕过该限制。
前轮严格 merged 仍为 13 PASS / 2 FAIL，本次未重跑、改写或将其计入通过项。
