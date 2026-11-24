# M3.1 BF16 基座 checkpoint 重载

本用例验证启用 LoRA 后更换**基座权重**，与 `/v1/load_lora_adapter` 更换 adapter 区分。
只测试 BF16、checkpoint-format 和默认 layerwise reload；在已有请求完成后重载。
本轮结果见 [重载验证报告](VALIDATION-reload-bf16.md)。

## 原理与门禁

从真实减层 checkpoint 另建目录，把 414 个语言侧 LoRA 目标对应的投影权重乘以 0.5，
并改变一个未被 LoRA 包装的 norm 参数作为对照。保留原始 checkpoint、形状和 dtype。
乘以 0.5 用于产生明确的权重变化，不代表训练更新或 merged LoRA 参考。

每组使用两个独立进程：第一个先加载原基座，重载新基座，再恢复原基座；第二个直接冷启动新基座。
比较所有 rank 上全部运行时参数的完整 SHA256、shape、dtype、非 meta 状态；
比较 3/17/129 token prompt、greedy 生成 8 token 的 base/A/B 原始 token ID 和 top-5 logprobs。
同时要求重载前后 adapter 缓冲区不变，恢复后的参数和输出完全一致。
新基座必须实际改变参数及至少一组输出；RPC 返回成功和 token 恰好相同均不足以判定通过。

## 准备新基座

沿用 [BF16 全模块验证](BF16_FULL_SCOPE.md) 的环境、4 层真实权重和 `all-a/all-b` 标准二维 adapter。
以下路径均使用绝对路径，`RUN` 必须是新目录；准备过程额外占用约一份减层 checkpoint 的磁盘空间。

```bash
TESTS=/vllm-workspace/vllm-ascend/tests/e2e/models/minimax_m31_lora
PYTHON=/path/to/uv-venv/bin/python
MODEL=/path/to/real4/base
ADAPTERS=/path/to/core
RUN=/path/to/new-reload-run
mkdir "$RUN"
export VLLM_WORKER_MULTIPROC_METHOD=spawn TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_BATCH_INVARIANT=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

"$PYTHON" "$TESTS/prepare_bf16_reload.py" \
  --model "$MODEL" --targets "$ADAPTERS/all-a/targets.json" \
  --control language_model.model.layers.0.input_layernorm.weight \
  --output "$RUN/changed-base"
```

`--control` 和 targets 中的逻辑名称须与实际 checkpoint 键一致；缺失目标会报错。
该示例使用 VL checkpoint 的语言侧，执行时设置 `language_model_only=True`。

## TP / EP 独立进程对照

从 `/workspace` 执行，示例保留单卡；TP2 时调整可见卡，并给两个 capture 同时添加 `--tp 2`；
TP2+EP2 时再同时添加 `--ep`。不要同时在相同设备上运行两个实例。

```bash
cd /workspace
export ASCEND_RT_VISIBLE_DEVICES=0
"$PYTHON" "$TESTS/validate_bf16_reload.py" capture \
  --model "$MODEL" --replacement "$RUN/changed-base" --lora \
  --adapters "$ADAPTERS/all-a" "$ADAPTERS/all-b" --output "$RUN/reloaded.json"
"$PYTHON" "$TESTS/validate_bf16_reload.py" capture \
  --model "$MODEL" --replacement "$RUN/changed-base" --lora --cold \
  --adapters "$ADAPTERS/all-a" "$ADAPTERS/all-b" --output "$RUN/cold.json"
"$PYTHON" "$TESTS/validate_bf16_reload.py" compare \
  --actual "$RUN/reloaded.json" --reference "$RUN/cold.json" --output "$RUN/comparison.json"
```

两个 capture 去掉 `--adapters` 可验证 LoRA 开启但未加载 adapter；同时去掉 `--lora` 可验证普通模型。
固定 KV cache 为 512 MiB，为 layerwise reload 临时张量预留空间；这不是生产容量或性能验收。

## TP2+DP2+EP4 HTTP 对照

服务仅监听 loopback。现有开发 RPC 用于受控测试，测试完成后停止该服务。

```bash
cd /workspace
VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 "$PYTHON" "$TESTS/serve_bf16.py" \
  --model "$MODEL" --adapters "$ADAPTERS" --tp 2 --dp 2 --ep \
  --kv-cache-memory-bytes 536870912 --worker-extension-cls reload_observation.ReloadWorker
```

另一个终端采集重载及恢复结果：

```bash
"$PYTHON" "$TESTS/validate_bf16_reload.py" capture-http \
  --model "$MODEL" --replacement "$RUN/changed-base" --tp 2 --dp 2 --ep \
  --output "$RUN/http-reloaded.json"
```

完成后停止此服务，用相同命令将 `--model` 改为 `"$RUN/changed-base"`，独立启动新服务；
客户端加 `--cold`，输出另存为 `http-cold.json`。最后使用相同 `compare` 子命令比较两份报告。
可在原基座恢复后、停止服务前，再运行 `validate_bf16_http.py` 检查真实 base/A/B 同批及 adapter C 的槽位复用。

配套 vLLM 的 HTTP `collective_rpc` 会在所有 DP engine 执行，但只返回第一个 DP engine 的结果。
`m31_reload_snapshot_all_dp` 因此使用已有 DP CPU group 汇总指纹，并带 DP/TP 身份；
客户端严格检查收到 `TP×DP` 份快照，没有降低 rank 覆盖门禁。
